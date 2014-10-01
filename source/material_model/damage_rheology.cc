/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/material_model/damage_rheology.h>
#include <deal.II/base/parameter_handler.h>

#include <iostream>

using namespace dealii;

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    double
    DamageRheology<dim>::
    phase_function (const Point<dim> &position,
                    const double temperature,
                    const double pressure,
                    const unsigned int phase) const
    {
      Assert(phase < transition_depths.size(),
          ExcMessage("Error: Phase index is too large. This phase index does not exist!"));

      // if we already have the adiabatic conditions, we can use them
      if (this->get_adiabatic_conditions().is_initialized())
        {
          // first, get the pressure at which the phase transition occurs normally
          const Point<dim,double> transition_point = this->get_geometry_model().representative_point(transition_depths[phase]);
          const double transition_pressure = this->get_adiabatic_conditions().pressure(transition_point);

          // then calculate the deviation from the transition point (both in temperature
          // and in pressure)
          double pressure_deviation = pressure - transition_pressure
                                      - transition_slopes[phase] * (temperature - transition_temperatures[phase]);

          // last, calculate the percentage of material that has undergone the transition
          return (pressure_deviation > 0) ? 1 : 0;
        }

      // if we do not have the adiabatic conditions, we have to use the depth instead
      // this is less precise, because we do not have the exact pressure gradient, instead we use pressure/depth
      // (this is for calculating e.g. the density in the adiabatic profile)
      else
        {
          double depth = this->get_geometry_model().depth(position);
          double depth_deviation = (pressure > 0
                                    ?
                                    depth - transition_depths[phase]
                                    - transition_slopes[phase] * (depth / pressure) * (temperature - transition_temperatures[phase])
                                    :
                                    depth - transition_depths[phase]
                                    - transition_slopes[phase] / (this->get_gravity_model().gravity_vector(position).norm() * reference_rho)
                                    * (temperature - transition_temperatures[phase]));

          return (depth_deviation > 0) ? 1 : 0;
        }
    }


    template <int dim>
    unsigned int
    DamageRheology<dim>::
    thermodynamic_phase (const double temperature,
                         const double pressure,
                         const std::vector<double> & /*composition*/) const
    {
      unsigned int phase_index = 0;
      for (unsigned int phase=0;phase<transition_depths.size();phase++)
        {
          // first, get the pressure at which the phase transition occurs normally
          const Point<dim,double> transition_point = this->get_geometry_model().representative_point(transition_depths[phase]);
          const double transition_pressure = this->get_adiabatic_conditions().pressure(transition_point);

          // then calculate the deviation from the transition point (both in temperature
          // and in pressure)
          double pressure_deviation = pressure - transition_pressure
                                      - transition_slopes[phase] * (temperature - transition_temperatures[phase]);

          // last, calculate the percentage of material that has undergone the transition
          if (pressure_deviation > 0)
            phase_index = phase+1;
        }

      return phase_index;
    }


    template <int dim>
    unsigned int
    DamageRheology<dim>::
    get_phase_index (const Point<dim> &position,
                     const double temperature,
                     const double pressure) const
    {
      Assert(grain_growth_activation_energy.size()>0,
                  ExcMessage("Error: No grain evolution parameters are given!"));

      unsigned int phase_index = 0;
      if(transition_depths.size()>0)
        if(phase_function(position, temperature, pressure, transition_depths.size()-1) == 1)
          phase_index = transition_depths.size();

      for(unsigned int j=1;j<transition_depths.size();++j)
        if(phase_function(position, temperature, pressure, j) != phase_function(position, temperature, pressure, j-1))
          phase_index = j;

      return phase_index;
    }

    template <int dim>
    double
    DamageRheology<dim>::
    grain_size_growth_rate (const double                  temperature,
                            const double                  pressure,
                            const std::vector<double>    &compositional_fields,
                            const SymmetricTensor<2,dim> &strain_rate,
                            const Tensor<1,dim>          &velocity,
                            const Point<dim>             &position,
                            const unsigned int            field_index,
                            const int                     crossed_transition) const
    {
      // we want to iterate over the grain size evolution here, as we solve in fact an ordinary differential equation
      // and it is not correct to use the starting grain size (and introduces instabilities)
      const double original_grain_size = compositional_fields[field_index];
      if((original_grain_size != original_grain_size) || this->get_timestep() == 0.0
                                                      || original_grain_size < std::numeric_limits<double>::min())
        return 0.0;

      // set up the parameters for the sub-timestepping of grain size evolution
      std::vector<double> current_composition = compositional_fields;
      double grain_size = original_grain_size;
      double grain_size_change = 0.0;
      const double timestep = this->get_timestep();
      double grain_growth_timestep = 500 * 3600 * 24 * 365.25; // 500 yrs
      double time = 0;

      // find out in which phase we are
      const unsigned int ol_index = get_phase_index(position, temperature, pressure);

      do
        {
          time += grain_growth_timestep;

          if(timestep - time < 0)
            {
              grain_growth_timestep = timestep - (time - grain_growth_timestep);
              time = timestep;
            }

          // grain size growth due to Ostwald ripening
          const double m = grain_growth_exponent[ol_index];
          const double grain_size_growth = grain_growth_rate_constant[ol_index] / (m * pow(grain_size,m-1))
                                   * exp(- (grain_growth_activation_energy[ol_index] + pressure * grain_growth_activation_volume[ol_index])
                                       / (gas_constant * temperature))
                                       * grain_growth_timestep;

          // grain size reduction in dislocation creep regime
          const SymmetricTensor<2,dim> shear_strain_rate = strain_rate - 1./dim * trace(strain_rate) * unit_symmetric_tensor<dim>();
          double second_strain_rate_invariant = std::sqrt(std::abs(second_invariant(shear_strain_rate)));

          const double dislocation_strain_rate = second_strain_rate_invariant
              * viscosity(temperature, pressure, current_composition, strain_rate, position)
              / dislocation_viscosity(temperature, pressure, current_composition, strain_rate, position);

          double grain_size_reduction = 0.0;

          if (use_paleowattmeter)
            {
              // paleowattmeter: Austin and Evans (2007): Paleowattmeters: A scaling relation for dynamically recrystallized grain size. Geology 35, 343-346
              const double stress = 2.0 * second_strain_rate_invariant * viscosity(temperature, pressure, current_composition, strain_rate, position);
              grain_size_reduction = stress * boundary_area_change_work_fraction[ol_index] * dislocation_strain_rate * pow(grain_size,2)
              / (geometric_constant[ol_index] * grain_boundary_energy[ol_index])
              * grain_growth_timestep;
            }
          else
            {
              // paleopiezometer: Hall and Parmentier (2003): Influence of grain size evolution on convective instability. Geochem. Geophys. Geosyst., 4(3).
              grain_size_reduction = reciprocal_required_strain[ol_index] * dislocation_strain_rate * grain_size * grain_growth_timestep;
            }

          grain_size_change = grain_size_growth - grain_size_reduction;

          if ((grain_size_change / grain_size < 0.001 && grain_size_growth / grain_size < 0.1
            && grain_size_reduction / grain_size < 0.1) || grain_size == 0.0)
            grain_growth_timestep *= 2;
          else if (grain_size_change / grain_size > 0.1 || grain_size_growth / grain_size > 0.5
              || grain_size_reduction / grain_size > 0.5)
            {
              grain_size_change = 0.0;
              time -= grain_growth_timestep;
              grain_growth_timestep /= 2.0;
            }

          grain_size += grain_size_change;
          current_composition[field_index] = grain_size;

          if (grain_size < 0)
            {
            std::cout << "Grain size smaller 0:  " << grain_size << " ," << grain_size_growth
                << " ," << grain_size_reduction << ", timestep: " << grain_growth_timestep << "! \n ";
            break;
            }
        }
      while (time < timestep);

      // reduce grain size to recrystallized_grain_size when crossing phase transitions
      // if the distance in radial direction a grain moved compared to the last time step
      // is crossing a phase transition, reduce grain size

      // TODO: recrystallize first, and then do grain size growth/reduction for grains that crossed the transition
      // in dependence of the distance they have moved
      double phase_grain_size_reduction = 0.0;
      if (this->introspection().name_for_compositional_index(field_index) == "olivine_grain_size"
          &&
          this->get_timestep_number() > 0)
        {
          // check if material has crossed any phase transition, if yes, reset grain size
          if (crossed_transition != -1)
            if (recrystallized_grain_size[crossed_transition] > 0.0)
              phase_grain_size_reduction = grain_size - recrystallized_grain_size[crossed_transition];
        }
      else if (this->introspection().name_for_compositional_index(field_index) == "pyroxene_grain_size")
        {
          phase_grain_size_reduction = 0.0;
        }

      if (grain_size < 5.e-6)
        {
          std::cout << "Grain size is " << grain_size << "! It needs to be larger than 5e-6.\n";
          grain_size = 5e-6;
        }

//      if(!(grain_size - grain_size == 0))
//        std::cout << "Grain size change is not a number! It is " << grain_size << "! \n";
      return grain_size - original_grain_size - phase_grain_size_reduction;
    }

    template <int dim>
    double
    DamageRheology<dim>::
    diffusion_viscosity (const double                  temperature,
                         const double                  pressure,
                         const std::vector<double>    &composition,
                         const SymmetricTensor<2,dim> &strain_rate,
                         const Point<dim>             &position) const
    {
      const SymmetricTensor<2,dim> shear_strain_rate = strain_rate - 1./dim * trace(strain_rate) * unit_symmetric_tensor<dim>();
      const double second_strain_rate_invariant = std::sqrt(-second_invariant(shear_strain_rate));

      // TODO: make this more general, for more phases we have to average grain size somehow
      // TODO: default when field is not given & warning
      const std::string field_name = "olivine_grain_size";
      const double grain_size = this->introspection().compositional_name_exists(field_name)
                                ?
                                composition[this->introspection().compositional_index_for_name(field_name)]
                                :
                                0.0;

      // find out in which phase we are
      const unsigned int ol_index = get_phase_index(position, temperature, pressure);

      // TODO: we use the prefactors from Behn et al., 2009 as default values, but their laws use the strain rate
      // and we use the second invariant --> check if the prefactors should be changed
      double energy_term = exp((diffusion_activation_energy[ol_index] + diffusion_activation_volume[ol_index] * abs(pressure))
                         / (diffusion_creep_exponent[ol_index] * gas_constant * temperature));
      if (this->get_adiabatic_conditions().is_initialized())
        {
          const double adiabatic_energy_term
            = exp((diffusion_activation_energy[ol_index] + diffusion_activation_volume[ol_index] * abs(pressure))
              / (diffusion_creep_exponent[ol_index] * gas_constant * this->get_adiabatic_conditions().temperature(position)));

          const double temperature_dependence = energy_term / adiabatic_energy_term;
          if (temperature_dependence > max_temperature_dependence_of_eta)
            energy_term = adiabatic_energy_term * max_temperature_dependence_of_eta;
          if (temperature_dependence < 1.0 / max_temperature_dependence_of_eta)
            energy_term = adiabatic_energy_term / max_temperature_dependence_of_eta;
        }

      const double strain_rate_dependence = (1.0 - diffusion_creep_exponent[ol_index]) / diffusion_creep_exponent[ol_index];

      return pow(diffusion_creep_prefactor[ol_index],-1.0/diffusion_creep_exponent[ol_index])
             * std::pow(second_strain_rate_invariant,strain_rate_dependence)
             * pow(grain_size, diffusion_creep_grain_size_exponent[ol_index]/diffusion_creep_exponent[ol_index])
             * energy_term;
    }

    template <int dim>
    double
    DamageRheology<dim>::
    dislocation_viscosity (const double      temperature,
                           const double      pressure,
                           const std::vector<double> &,
                           const SymmetricTensor<2,dim> &strain_rate,
                           const Point<dim> &position) const
    {
      const SymmetricTensor<2,dim> shear_strain_rate = strain_rate - 1./dim * trace(strain_rate) * unit_symmetric_tensor<dim>();
      const double second_strain_rate_invariant = std::sqrt(-second_invariant(shear_strain_rate));

      // find out in which phase we are
      const unsigned int ol_index = get_phase_index(position, temperature, pressure);

      double energy_term = exp((dislocation_activation_energy[ol_index] + dislocation_activation_volume[ol_index] * pressure)
                         / (dislocation_creep_exponent[ol_index] * gas_constant * temperature));
      if (this->get_adiabatic_conditions().is_initialized())
        {
          const double adiabatic_energy_term
            = exp((dislocation_activation_energy[ol_index] + dislocation_activation_volume[ol_index] * pressure)
              / (dislocation_creep_exponent[ol_index] * gas_constant * this->get_adiabatic_conditions().temperature(position)));

          const double temperature_dependence = energy_term / adiabatic_energy_term;
          if (temperature_dependence > max_temperature_dependence_of_eta)
            energy_term = adiabatic_energy_term * max_temperature_dependence_of_eta;
          if (temperature_dependence < 1.0 / max_temperature_dependence_of_eta)
            energy_term = adiabatic_energy_term / max_temperature_dependence_of_eta;
        }

      const double strain_rate_dependence = (1.0 - dislocation_creep_exponent[ol_index]) / dislocation_creep_exponent[ol_index];

      return pow(dislocation_creep_prefactor[ol_index],-1.0/dislocation_creep_exponent[ol_index])
             * std::pow(second_strain_rate_invariant,strain_rate_dependence)
             * energy_term;
    }

    template <int dim>
    double
    DamageRheology<dim>::
    viscosity_ratio (const double temperature,
                     const double pressure,
                     const std::vector<double> &composition,
                     const SymmetricTensor<2,dim> &strain_rate,
                     const Point<dim> &position) const
    {
      return dislocation_viscosity(temperature,pressure,composition,strain_rate,position)
           / diffusion_viscosity(temperature,pressure,composition,strain_rate,position);
    }

    template <int dim>
    double
    DamageRheology<dim>::
    viscosity (const double temperature,
               const double pressure,
               const std::vector<double> &composition,
               const SymmetricTensor<2,dim> &strain_rate,
               const Point<dim> &position) const
    {
      //TODO: add assert
      /*if (this->get_timestep_number() > 0)
        Assert (grain_size >= 1.e-6, ExcMessage ("Error: The grain size should not be smaller than 1e-6 m."));*/

      const SymmetricTensor<2,dim> shear_strain_rate = strain_rate - 1./dim * trace(strain_rate) * unit_symmetric_tensor<dim>();
      const double second_strain_rate_invariant = std::sqrt(-second_invariant(shear_strain_rate));

      const double diff_viscosity = diffusion_viscosity(temperature, pressure, composition, strain_rate, position);
      const double disl_viscosity = dislocation_viscosity(temperature, pressure, composition, strain_rate, position);

      double effective_viscosity;
      if(std::abs(second_strain_rate_invariant) > 1e-30)//1e6*std::numeric_limits<double>::min())
        effective_viscosity = disl_viscosity * diff_viscosity / (disl_viscosity + diff_viscosity);
      else
        effective_viscosity = diff_viscosity;
      return effective_viscosity;
    }


    template <int dim>
    double
    DamageRheology<dim>::
    reference_viscosity () const
    {
      return eta;
    }

    template <int dim>
    double
    DamageRheology<dim>::
    reference_density () const
    {
      return reference_rho;
    }


    template <int dim>
    double
    DamageRheology<dim>::
    density (const double temperature,
             const double pressure,
             const std::vector<double> &compositional_fields, /*composition*/
             const Point<dim> &) const
    {
      const double composition_dependence = compositional_fields.size()>0
                                            ?
                                            compositional_delta_rho * compositional_fields[0]
                                            :
                                            0.0;

      return (reference_rho + composition_dependence) * std::exp(reference_compressibility * (pressure - this->get_surface_pressure()))
          * (1 - thermal_alpha * (temperature - reference_T));
    }


    template <int dim>
    bool
    DamageRheology<dim>::
    viscosity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      // compare this with the implementation of the viscosity() function
      // to see the dependencies
      if ((dependence & NonlinearDependence::temperature) != NonlinearDependence::none)
        return true;
      else if ((dependence & NonlinearDependence::compositional_fields) != NonlinearDependence::none)
        return true;
      else if((dependence & NonlinearDependence::strain_rate) != NonlinearDependence::none)
        return true;
      else if((dependence & NonlinearDependence::pressure) != NonlinearDependence::none)
        return true;
      else
        return false;
    }


    template <int dim>
    bool
    DamageRheology<dim>::
    density_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      // compare this with the implementation of the density() function
      // to see the dependencies
      if (((dependence & NonlinearDependence::temperature) != NonlinearDependence::none)
          &&
          (thermal_alpha != 0))
        return true;
      else
        return false;
    }

    template <int dim>
    bool
    DamageRheology<dim>::
    compressibility_depends_on (const NonlinearDependence::Dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    DamageRheology<dim>::
    specific_heat_depends_on (const NonlinearDependence::Dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    DamageRheology<dim>::
    thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      return false;
    }



    template <int dim>
    bool
    DamageRheology<dim>::
    is_compressible () const
    {
      return (reference_compressibility != 0);
    }


    template <int dim>
    void
    DamageRheology<dim>::
    evaluate(const typename Interface<dim>::MaterialModelInputs &in, typename Interface<dim>::MaterialModelOutputs &out) const
    {
      for (unsigned int i=0; i<in.position.size(); ++i)
        {
          // set up an integer that tells us which phase transition has been crossed inside of the cell
          int crossed_transition(-1);

          if (this->get_adiabatic_conditions().is_initialized())
            for (unsigned int phase=0;phase<transition_depths.size();++phase)
              {
                // first, get the pressure at which the phase transition occurs normally
                const Point<dim,double> transition_point = this->get_geometry_model().representative_point(transition_depths[phase]);
                const Point<dim,double> transition_plus_width = this->get_geometry_model().representative_point(transition_depths[phase] + transition_widths[phase]);
                const Point<dim,double> transition_minus_width = this->get_geometry_model().representative_point(transition_depths[phase] - transition_widths[phase]);
                const double transition_pressure = this->get_adiabatic_conditions().pressure(transition_point);
                const double pressure_width = 0.5 * (this->get_adiabatic_conditions().pressure(transition_plus_width)
                                                     - this->get_adiabatic_conditions().pressure(transition_minus_width));


                // then calculate the deviation from the transition point (both in temperature
                // and in pressure)
                double pressure_deviation = in.pressure[i] - transition_pressure
                                            - transition_slopes[phase] * (in.temperature[i] - transition_temperatures[phase]);

                if ((std::abs(pressure_deviation) < pressure_width)
                  &&
                  ((in.velocity[i] * this->get_gravity_model().gravity_vector(in.position[i])) * pressure_deviation > 0))
                  crossed_transition = phase;
              }
          else
            for (unsigned int j=0; j<in.position.size(); ++j)
              for (unsigned int k=0;k<transition_depths.size();++k)
                if((phase_function(in.position[i], in.temperature[i], in.pressure[i], k)
                    != phase_function(in.position[j], in.temperature[j], in.pressure[j], k))
                    &&
                    ((in.velocity[i] * this->get_gravity_model().gravity_vector(in.position[i]))
                    * ((in.position[i] - in.position[j]) * this->get_gravity_model().gravity_vector(in.position[i])) > 0))
                  crossed_transition = k;

          if (in.strain_rate.size() > 0)
            out.viscosities[i] = std::min(std::max(eta*1.e-5,viscosity(in.temperature[i],
                                                                       in.pressure[i],
                                                                       in.composition[i],
                                                                       in.strain_rate[i],
                                                                       in.position[i])),eta*1.e5);

          out.densities[i] = density(in.temperature[i], in.pressure[i], in.composition[i], in.position[i]);
          out.thermal_expansion_coefficients[i] = thermal_alpha;
          out.specific_heat[i] = reference_specific_heat;
          out.thermal_conductivities[i] = k_value;
          out.compressibilities[i] = reference_compressibility;

          // TODO: make this more general for not just olivine grains
          if (in.strain_rate.size() > 0)
            for (unsigned int c=0;c<in.composition[i].size();++c)
              {
                if (this->introspection().name_for_compositional_index(c) == "olivine_grain_size")
                  out.reaction_terms[i][c] = grain_size_growth_rate(in.temperature[i], in.pressure[i], in.composition[i],
                      in.strain_rate[i], in.velocity[i], in.position[i], c, crossed_transition);
                else
                  out.reaction_terms[i][c] = 0.0;
              }
        }

    }


    template <int dim>
    void
    DamageRheology<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Damage rheology model");
        {
          prm.declare_entry ("Reference density", "3300",
                             Patterns::Double (0),
                             "Reference density $\\rho_0$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference temperature", "293",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. Units: $K$.");
          prm.declare_entry ("Viscosity", "5e24",
                             Patterns::Double (0),
                             "The value of the constant viscosity. Units: $kg/m/s$.");
          prm.declare_entry ("Composition viscosity prefactor 1", "1.0",
                             Patterns::Double (0),
                             "A linear dependency of viscosity on the first compositional field. "
                             "Dimensionless prefactor. With a value of 1.0 (the default) the "
                             "viscosity does not depend on the composition.");
          prm.declare_entry ("Composition viscosity prefactor 2", "1.0",
                             Patterns::Double (0),
                             "A linear dependency of viscosity on the second compositional field. "
                             "Dimensionless prefactor. With a value of 1.0 (the default) the "
                             "viscosity does not depend on the composition.");
          prm.declare_entry ("Compositional density difference", "100.0",
                             Patterns::Double (),
                             "Density excess of the first compositional field."
                             "Units: $kg/m^3$");
          prm.declare_entry ("Thermal conductivity", "4.7",
                             Patterns::Double (0),
                             "The value of the thermal conductivity $k$. "
                             "Units: $W/m/K$.");
          prm.declare_entry ("Reference specific heat", "1250",
                             Patterns::Double (0),
                             "The value of the specific heat $cp$. "
                             "Units: $J/kg/K$.");
          prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");
          prm.declare_entry ("Reference compressibility", "4e-12",
                             Patterns::Double (0),
                             "The value of the reference compressibility. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Phase transition depths", "",
                             Patterns::List (Patterns::Double(0)),
                             "A list of depths where phase transitions occur. Values must "
                             "monotonically increase. "
                             "Units: $m$.");
          prm.declare_entry ("Phase transition temperatures", "",
                             Patterns::List (Patterns::Double(0)),
                             "A list of temperatures where phase transitions occur. Higher or lower "
                             "temperatures lead to phase transition ocurring in smaller or greater "
                             "depths than given in Phase transition depths, depending on the "
                             "Clapeyron slope given in Phase transition Clapeyron slopes. "
                             "List must have the same number of entries as Phase transition depths. "
                             "Units: $K$.");
          prm.declare_entry ("Phase transition widths", "",
                             Patterns::List (Patterns::Double(0)),
                             "A list of widths for each phase transition. This is only use to specify "
                             "the region where the recrystallized grain size is assigned after material "
                             "has crossed a phase transition and should accordingly be chosen similar "
                             "to the maximum cell width expected at the phase transition."
                             "List must have the same number of entries as Phase transition depths. "
                             "Units: $m$.");
          prm.declare_entry ("Phase transition Clapeyron slopes", "",
                             Patterns::List (Patterns::Double()),
                             "A list of Clapeyron slopes for each phase transition. A positive "
                             "Clapeyron slope indicates that the phase transition will occur in "
                             "a greater depth, if the temperature is higher than the one given in "
                             "Phase transition temperatures and in a smaller depth, if the "
                             "temperature is smaller than the one given in Phase transition temperatures. "
                             "For negative slopes the other way round. "
                             "List must have the same number of entries as Phase transition depths. "
                             "Units: $Pa/K$.");
          prm.declare_entry ("Corresponding phase for transition", "",
                             Patterns::List(Patterns::Anything()),
                             "A user-defined list of phases, which correspond to the name of the phase the "
                             "transition should occur in. "
                             "List must have the same number of entries as Phase transition depths. "
                             "Units: $Pa/K$.");
          prm.declare_entry ("Grain growth activation energy", "3.5e5",
                             Patterns::List (Patterns::Double(0)),
                             "The activation energy for grain growth $E_g$. "
                             "Units: $J/mol$.");
          prm.declare_entry ("Grain growth activation volume", "8e-6",
                             Patterns::List (Patterns::Double(0)),
                             "The activation volume for grain growth $E_g$. "
                             "Units: $m^3/mol$.");
          prm.declare_entry ("Grain growth exponent", "3",
                             Patterns::List (Patterns::Double(0)),
                             "Exponent of the grain growth law $p_g$. This is an experimentally determined "
                             "grain growth constant. "
                             "Units: none.");
          prm.declare_entry ("Grain growth rate constant", "1.5e-5",
                             Patterns::List (Patterns::Double(0)),
                             "Prefactor of the Ostwald ripening grain growth law $G_0$. "
                             "This is dependent on water content, which is assumed to be "
                             "50 H/10^6 Si for the default value. "
                             "Units: $m^{p_g}/s$.");
          prm.declare_entry ("Reciprocal required strain", "10",
                             Patterns::List (Patterns::Double(0)),
                             "This parameters $\\lambda$ gives an estimate of the strain necessary "
                             "to achieve a new grain size. ");
          prm.declare_entry ("Recrystallized grain size", "0.001",
                             Patterns::List (Patterns::Double(0)),
                             "The grain size $d_{ph}$ to that a phase will be reduced to when crossing a phase transition. "
                             "When set to zero, grain size will not be reduced. "
                             "Units: m.");
          prm.declare_entry ("Use paleowattmeter", "true",
                             Patterns::Bool (),
                             "A flag indicating whether the computation should be use the "
                             "paleowattmeter approach of Austin and Evans (2007) for grain size reduction "
                             "in the dislocation creep regime (if true) or the paleopiezometer aprroach "
                             "from Hall and Parmetier (2003) (if false).");
          prm.declare_entry ("Average specific grain boundary energy", "1.0",
                             Patterns::List (Patterns::Double(0)),
                             "The average specific grain boundary energy $\\gamma$. "
                             "Units: J/m^2.");
          prm.declare_entry ("Work fraction for boundary area change", "0.1",
                             Patterns::List (Patterns::Double(0)),
                             "The fraction $\\chi$ of work done by dislocation creep to change the grain boundary area. "
                             "Units: J/m^2.");
          prm.declare_entry ("Geometric constant", "3",
                             Patterns::List (Patterns::Double(0)),
                             "Geometric constant $c$ used in the paleowattmeter grain size reduction law. "
                             "Units: none.");
          prm.declare_entry ("Dislocation creep exponent", "3.5",
                             Patterns::List (Patterns::Double(0)),
                             "Power-law exponent $n_{dis}$ for dislocation creep. "
                             "Units: none.");
          prm.declare_entry ("Dislocation activation energy", "4.8e5",
                             Patterns::List (Patterns::Double(0)),
                             "The activation energy for dislocation creep $E_{dis}$. "
                             "Units: $J/mol$.");
          prm.declare_entry ("Dislocation activation volume", "1.1e-5",
                             Patterns::List (Patterns::Double(0)),
                             "The activation volume for dislocation creep $V_{dis}$. "
                             "Units: $m^3/mol$.");
          prm.declare_entry ("Dislocation creep prefactor", "4.5e-15",
                             Patterns::List (Patterns::Double(0)),
                             "Prefactor for the dislocation creep law $A_{dis}$. "
                             "Units: $Pa^{-n_{dis}}/s$.");
          prm.declare_entry ("Diffusion creep exponent", "1",
                             Patterns::List (Patterns::Double(0)),
                             "Power-law exponent $n_{diff}$ for diffusion creep. "
                             "Units: none.");
          prm.declare_entry ("Diffusion activation energy", "3.35e5",
                             Patterns::List (Patterns::Double(0)),
                             "The activation energy for diffusion creep $E_{diff}$. "
                             "Units: $J/mol$.");
          prm.declare_entry ("Diffusion activation volume", "4e-6",
                             Patterns::List (Patterns::Double(0)),
                             "The activation volume for diffusion creep $V_{diff}$. "
                             "Units: $m^3/mol$.");
          prm.declare_entry ("Diffusion creep prefactor", "7.4e-15",
                             Patterns::List (Patterns::Double(0)),
                             "Prefactor for the diffusion creep law $A_{diff}$. "
                             "Units: $m^{p_{diff}} Pa^{-n_{diff}}/s$.");
          prm.declare_entry ("Diffusion creep grain size exponent", "3",
                             Patterns::List (Patterns::Double(0)),
                             "Diffusion creep grain size exponent $p_{diff}$ that determines the "
                             "dependence of vescosity on grain size. "
                             "Units: none.");
          prm.declare_entry ("Maximum temperature dependence of viscosity", "100",
                             Patterns::Double (0),
                             "The factor by which viscosity at adiabatic temperature and ambient temperature "
                             "are allowed to differ (a value of x means that the viscosity can be x times higher "
                             "or x times lower compared to the value at adiabatic temperature. This parameter "
                             "is introduced to limit local viscosity contrasts, but still allow for a widely "
                             "varying viscosity over the wole mantle range. "
                             "Units: none.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    DamageRheology<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Damage rheology model");
        {
          gas_constant               = 8.314462;
          reference_rho              = prm.get_double ("Reference density");
          reference_T                = prm.get_double ("Reference temperature");
          eta                        = prm.get_double ("Viscosity");
          composition_viscosity_prefactor_1 = prm.get_double ("Composition viscosity prefactor 1");
          composition_viscosity_prefactor_2 = prm.get_double ("Composition viscosity prefactor 2");
          compositional_delta_rho = prm.get_double("Compositional density difference");
          k_value                    = prm.get_double ("Thermal conductivity");
          reference_specific_heat    = prm.get_double ("Reference specific heat");
          thermal_alpha              = prm.get_double ("Thermal expansion coefficient");
          reference_compressibility  = prm.get_double ("Reference compressibility");


          transition_depths         = Utilities::string_to_double
                                      (Utilities::split_string_list(prm.get ("Phase transition depths")));
          transition_temperatures   = Utilities::string_to_double
                                      (Utilities::split_string_list(prm.get ("Phase transition temperatures")));
          transition_slopes         = Utilities::string_to_double
                                      (Utilities::split_string_list(prm.get ("Phase transition Clapeyron slopes")));
          transition_phases         = Utilities::split_string_list (prm.get("Corresponding phase for transition"));
          recrystallized_grain_size = Utilities::string_to_double
                                      (Utilities::split_string_list(prm.get ("Recrystallized grain size")));
          transition_widths         = Utilities::string_to_double
                                      (Utilities::split_string_list(prm.get ("Phase transition widths")));

          if (transition_temperatures.size() != transition_depths.size() ||
              transition_slopes.size() != transition_depths.size() ||
              transition_phases.size() != transition_depths.size() ||
              transition_widths.size() != transition_depths.size() ||
              recrystallized_grain_size.size() != transition_depths.size() )
            Assert(false,
                ExcMessage("Error: At least one list that gives input parameters for the phase transitions has the wrong size."));

          // TODO: add assert that transition depths are in increasing order

          // grain evolution parameters
          grain_growth_activation_energy        = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Grain growth activation energy")));
          grain_growth_activation_volume        = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Grain growth activation volume")));
          grain_growth_rate_constant            = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Grain growth rate constant")));
          grain_growth_exponent                 = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Grain growth exponent")));
          reciprocal_required_strain            = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Reciprocal required strain")));

          use_paleowattmeter                    = prm.get_bool ("Use paleowattmeter");
          grain_boundary_energy                 = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Average specific grain boundary energy")));
          boundary_area_change_work_fraction    = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Work fraction for boundary area change")));
          geometric_constant                    = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Geometric constant")));

          // rheology parameters
          dislocation_creep_exponent            = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Dislocation creep exponent")));
          dislocation_activation_energy         = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Dislocation activation energy")));
          dislocation_activation_volume         = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Dislocation activation volume")));
          dislocation_creep_prefactor           = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Dislocation creep prefactor")));
          diffusion_creep_exponent              = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Diffusion creep exponent")));
          diffusion_activation_energy           = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Diffusion activation energy")));
          diffusion_activation_volume           = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Diffusion activation volume")));
          diffusion_creep_prefactor             = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Diffusion creep prefactor")));
          diffusion_creep_grain_size_exponent   = Utilities::string_to_double
                                                  (Utilities::split_string_list(prm.get ("Diffusion creep grain size exponent")));
          max_temperature_dependence_of_eta     = prm.get_double ("Maximum temperature dependence of viscosity");

          if (grain_growth_activation_energy.size() != grain_growth_activation_volume.size() ||
              grain_growth_activation_energy.size() != grain_growth_rate_constant.size() ||
              grain_growth_activation_energy.size() != grain_growth_exponent.size() ||
              grain_growth_activation_energy.size() != dislocation_creep_exponent.size() ||
              grain_growth_activation_energy.size() != dislocation_activation_energy.size() ||
              grain_growth_activation_energy.size() != dislocation_activation_volume.size() ||
              grain_growth_activation_energy.size() != dislocation_creep_prefactor.size() ||
              grain_growth_activation_energy.size() != diffusion_creep_exponent.size() ||
              grain_growth_activation_energy.size() != diffusion_activation_energy.size() ||
              grain_growth_activation_energy.size() != diffusion_activation_volume.size() ||
              grain_growth_activation_energy.size() != diffusion_creep_prefactor.size() ||
              grain_growth_activation_energy.size() != diffusion_creep_grain_size_exponent.size() )
              Assert(false,
                ExcMessage("Error: The lists of grain size evolution and flow law parameters "
                           "need to have the same length!"));

          if(use_paleowattmeter)
            {
              if(grain_growth_activation_energy.size() != grain_boundary_energy.size() ||
                 grain_growth_activation_energy.size() != boundary_area_change_work_fraction.size() ||
                 grain_growth_activation_energy.size() != geometric_constant.size() )
                Assert(false,
                ExcMessage("Error: One of the lists of grain size evolution parameters "
                           "given for the paleowattmeter does not have the correct length!"));
            }
          else
              Assert(grain_growth_activation_energy.size() == reciprocal_required_strain.size(),
              ExcMessage("Error: The list of grain size evolution parameters in the "
                         "paleopiezometer does not have the correct length!"));

          Assert(grain_growth_activation_energy.size() == transition_depths.size()+1,
          ExcMessage("Error: The lists of grain size evolution and flow law parameters need to "
                     "have exactly one more entry than the number of phase transitions "
                     "(which is defined by the length of the lists of phase transition depths, ...)!"));


        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(DamageRheology,
                                   "damage rheology",
                                   "A material model that behaves in the same way as "
                                   "the simple material model, but includes compositional "
                                   "fields that stand for average grain sizes of a mineral "
                                   "phase and source terms for them that determine the grain "
                                   "size evolution in dependence of the strain rate, "
                                   "temperature, phase transitions, and the creep regime. "
                                   "In the diffusion creep regime, the viscosity depends "
                                   "on this grain size."
                                   "We use the grain size evolution laws described in Behn "
                                   "et al., 2009. Implications of grain size evolution on the "
                                   "seismic structure of the oceanic upper mantle, "
                                   "Earth Planet. Sci. Letters, 282, 178–189.")
  }
}
