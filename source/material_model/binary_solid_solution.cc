/*
  Copyright (C) 2015 - 2017 by the authors of the ASPECT code.

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
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/material_model/binary_solid_solution.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/table.h>
#include <fstream>
#include <iostream>


namespace aspect
{
  namespace MaterialModel
  {

    template <int dim>
    double
    BinarySolidSolution<dim>::
    reference_viscosity () const
    {
      return eta_0;
    }

    template <int dim>
    double
    BinarySolidSolution<dim>::
    reference_darcy_coefficient () const
    {
      // 0.01 = 1% melt
      return reference_permeability * std::pow(0.01,3.0) / eta_f;
    }

    template <int dim>
    bool
    BinarySolidSolution<dim>::
    is_compressible () const
    {
      return false;
    }


    template <int dim>
    double
    BinarySolidSolution<dim>::
    limit_update_to_0_and_1 (const double old_value,
                             const double change_of_value) const
    {
      if (old_value + change_of_value < 0)
        return -old_value;
      else if (old_value + change_of_value > 1)
        return 1.0 - old_value;
      else
        return change_of_value;
    }


    template <int dim>
    double
    BinarySolidSolution<dim>::
    melt_fraction (const double temperature,
                   const double pressure,
                   const double bulk_composition,
                   double &new_solid_composition,
                   double &new_melt_composition) const
    {
      // after Phipps Morgan, Jason. "Thermodynamics of pressure release melting of a veined plum pudding mantle."
      // Geochemistry, Geophysics, Geosystems 2.4 (2001). Values below taken from table A1.

      const double P = pressure;                   // pressure in Pa
      const double T = temperature;                // tmperature in K
      const double R = constants::gas_constant;    // Ideal Gas Constant

      const double T_Fo_melting_surface = 2163.0;  // Kelvin at 1 atmosphere - reference melting temperature for forsterite
      const double T_Fa_melting_surface = 1478.0;  // Kelvin at 1 atmosphere - reference melting temperature for fayalite

      const double dS_Fo = 60.0;                   // entropy change of melting in J/mol K = 901.961 J/(kg K)
      const double dS_Fa = 60.0;                   // entropy change of melting in J/mol K = 623.025 J/(kg K)

      const double vliq_ref = 49.0/1.e6;           // reference molar volume of the melt in m3/mol
      const double vsol_ref_Fo = 46.0/1.e6;        // reference molar volume of solid forsterite in m3/mol
      const double vsol_ref_Fa = 46.0/1.e6;        // reference molar volume of solid fayalite in m3/mol

      const double surface_pressure = 1.e5;        // 1 atmosphere = 1e5 Pa

      // Free Energy Change Delta_G due to Melting as a function of temperature and pressure, for forsterite and fayalite.
      // Equation (A9) in Phipps Morgan (2001).
      const double dG_Fo = (T_Fo_melting_surface - T) * dS_Fo
                               + (vliq_ref - vsol_ref_Fo) * (P-surface_pressure)
                               - ((vliq_ref * melt_compressibility - vsol_ref_Fo * forsterite_compressibility) * 0.5 * (std::pow(P,2) - std::pow(surface_pressure,2)));
      const double dG_Fa = (T_Fa_melting_surface - T) * dS_Fa
                               + (vliq_ref - vsol_ref_Fa) * (P-surface_pressure)
                               - ((vliq_ref * melt_compressibility - vsol_ref_Fa * fayalite_compressibility) * 0.5 * (std::pow(P,2) - std::pow(surface_pressure,2)));

      // Mole Fraction of Each Component in Coexisting Liquids and Solids, equations (A10 - A12) in Phipps Morgan (2001)
      double Xl_Fo = 1.0 - (1.0 - exp(dG_Fo/(2.0*R*T)))/(exp(dG_Fa/(2.0*R*T))-exp(dG_Fo/(2.0*R*T)));
      double Xs_Fo = Xl_Fo * exp(dG_Fo/(2.0*R*T));

      double melt_fraction;
      if (Xs_Fo <= bulk_composition)      // below the solidus
        {
          melt_fraction = 0;
          new_solid_composition = bulk_composition;
        }
      else if (Xl_Fo >= bulk_composition) // above the liquidus
        {
          melt_fraction = 1;
          new_melt_composition = bulk_composition;
        }
      else                                // between solidus and liquidis
        {
          new_solid_composition = Xs_Fo;
          new_melt_composition = Xl_Fo;
          melt_fraction = (bulk_composition - Xs_Fo)/(Xl_Fo - Xs_Fo);
        }

      return melt_fraction;
    }


    template <int dim>
    void
    BinarySolidSolution<dim>::
    melt_fractions (const MaterialModel::MaterialModelInputs<dim> &in,
                    std::vector<double> &melt_fractions) const
    {
      double bulk_composition = 0.89;
      double solid_composition = 0.89;
      double melt_composition = 0.89;

      for (unsigned int q=0; q<in.temperature.size(); ++q)
        {
          if (this->introspection().compositional_name_exists("solid_composition")
              && this->introspection().compositional_name_exists("melt_composition")
              && this->introspection().compositional_name_exists("porosity"))
            {
              const double porosity = in.composition[q][this->introspection().compositional_index_for_name("porosity")];
              solid_composition = in.composition[q][this->introspection().compositional_index_for_name("solid_composition")];
              melt_composition = in.composition[q][this->introspection().compositional_index_for_name("melt_composition")];
              bulk_composition = porosity * melt_composition + (1.0 - porosity) * solid_composition;
            }
          melt_fractions[q] = this->melt_fraction(in.temperature[q],
                                                  std::max(0.0, in.pressure[q]),
                                                  bulk_composition,
                                                  solid_composition,
                                                  melt_composition);
        }
      return;
    }


    template <int dim>
    void
    BinarySolidSolution<dim>::
    evaluate(const typename Interface<dim>::MaterialModelInputs &in, typename Interface<dim>::MaterialModelOutputs &out) const
    {
      ReactionRateOutputs<dim> *reaction_rate_out = out.template get_additional_output<ReactionRateOutputs<dim> >();

      if (this->include_melt_transport() && include_melting_and_freezing)
       AssertThrow(this->get_parameters().use_operator_splitting && this->get_parameters().reaction_steps_per_advection_step == 1,
                    ExcMessage("Material model binary solid solution with melt transport only "
                               "works with the Operator splitting solver option enabled, "
                               "and the Reaction time steps per advection step have to be set to 1."));

      // make sure the compositional fields we want to use exist
      if (this->include_melt_transport())
        AssertThrow(this->introspection().compositional_name_exists("porosity"),
                    ExcMessage("Material model binary solid solution with melt transport only "
                               "works if there is a compositional field called porosity."));

      if (this->include_melt_transport() && include_melting_and_freezing)
        AssertThrow(this->introspection().compositional_name_exists("solid_composition"),
                    ExcMessage("Material model binary solid solution with melting and freezing "
                               "only works if there is a compositional field called "
                               "solid_composition."));

      if (this->include_melt_transport() && include_melting_and_freezing)
        AssertThrow(this->introspection().compositional_name_exists("melt_composition"),
                    ExcMessage("Material model binary solid solution with melting and freezing "
                               "only works if there is a compositional field called "
                               "melt_composition."));

      for (unsigned int i=0; i<in.position.size(); ++i)
        {
          // calculate density first, we need it for the reaction term
          // temperature dependence of density is 1 - alpha * (T - T(adiabatic))
          double temperature_dependence = 1.0;
          if (this->include_adiabatic_heating ())
            temperature_dependence -= (in.temperature[i] - this->get_adiabatic_conditions().temperature(in.position[i]))
                                      * thermal_expansivity;
          else
            temperature_dependence -= (in.temperature[i] - reference_T) * thermal_expansivity;

          // calculate composition dependence of density
          const double fo_density = forsterite_density * std::exp(forsterite_compressibility * (in.pressure[i] - this->get_surface_pressure()));
          const double fa_density = fayalite_density * std::exp(fayalite_compressibility * (in.pressure[i] - this->get_surface_pressure()));
          const double fo_fraction = this->introspection().compositional_name_exists("solid_composition")
                                     ?
                                     std::min(1.0, std::max(0.0, in.composition[i][this->introspection().compositional_index_for_name("solid_composition")]))
                                     :
                                     1.0;

          out.densities[i] = (fo_density * fo_fraction + fa_density * (1.0 -  fo_fraction)) * temperature_dependence;

          // now compute melting and crystallization
          if (this->include_melt_transport() && include_melting_and_freezing)
            {
              const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
              const unsigned int solid_composition_idx = this->introspection().compositional_index_for_name("solid_composition");
              const unsigned int melt_composition_idx = this->introspection().compositional_index_for_name("melt_composition");

              const double old_porosity = in.composition[i][porosity_idx];
              const double old_solid_composition = in.composition[i][solid_composition_idx];
              const double old_melt_composition = in.composition[i][melt_composition_idx];

              const double porosity = std::min(1.0, std::max(old_porosity,0.0));
              double solid_composition = std::min(1.0, std::max(old_solid_composition,0.0));
              double melt_composition = std::min(1.0, std::max(old_melt_composition,0.0));

              const double bulk_composition = porosity * melt_composition + (1.0 - porosity) * solid_composition;

              // calculate the melting rate as difference between the equilibrium melt fraction
              // and the solution of the previous time step, and also update melt and solid composition
              const double eq_melt_fraction = melt_fraction(in.temperature[i],
                                                            this->get_adiabatic_conditions().pressure(in.position[i]),
                                                            bulk_composition,
                                                            solid_composition,
                                                            melt_composition);

              // do not allow negative porosity or porosity > 1
              const double melting_rate = limit_update_to_0_and_1(old_porosity, eq_melt_fraction - old_porosity);
              const double change_of_solid_composition = limit_update_to_0_and_1(old_solid_composition, solid_composition - old_solid_composition);
              const double change_of_melt_composition = limit_update_to_0_and_1(old_melt_composition, melt_composition - old_melt_composition);

              for (unsigned int c=0; c<in.composition[i].size(); ++c)
                {
                  // fill reaction rate outputs
                  if (reaction_rate_out != NULL)
                    {
                      if (c == solid_composition_idx && this->get_timestep_number() > 0)
                        reaction_rate_out->reaction_rates[i][c] = change_of_solid_composition / this->get_timestep()
                                                                  - old_solid_composition * trace(in.strain_rate[i]);
                      else if (c == melt_composition_idx && this->get_timestep_number() > 0)
                        reaction_rate_out->reaction_rates[i][c] = change_of_melt_composition / this->get_timestep();
                      else if (c == porosity_idx && this->get_timestep_number() > 0)
                        reaction_rate_out->reaction_rates[i][c] = melting_rate / this->get_timestep();
                      else
                        reaction_rate_out->reaction_rates[i][c] = 0.0;
                    }
                  out.reaction_terms[i][c] = 0.0;
                }

              out.viscosities[i] = eta_0 * exp(- alpha_phi * porosity);
            }
          else
            {
              out.viscosities[i] = eta_0;
              for (unsigned int c=0; c<in.composition[i].size(); ++c)
                out.reaction_terms[i][c] = 0.0;
            }

          out.entropy_derivative_pressure[i]    = 0.0;
          out.entropy_derivative_temperature[i] = 0.0;
          out.thermal_expansion_coefficients[i] = thermal_expansivity;
          out.specific_heat[i] = reference_specific_heat;
          out.thermal_conductivities[i] = thermal_conductivity;
          out.compressibilities[i] = 0.0;

          double visc_temperature_dependence = 1.0;
          if (this->include_adiabatic_heating ())
            {
              const double delta_temp = in.temperature[i]-this->get_adiabatic_conditions().temperature(in.position[i]);
              visc_temperature_dependence = std::max(std::min(std::exp(-thermal_viscosity_exponent*delta_temp/this->get_adiabatic_conditions().temperature(in.position[i])),1e4),1e-4);
            }
          else
            {
              const double delta_temp = in.temperature[i]-reference_T;
              visc_temperature_dependence = std::max(std::min(std::exp(-thermal_viscosity_exponent*delta_temp/reference_T),1e4),1e-4);
            }
          out.viscosities[i] *= visc_temperature_dependence;
        }

      // fill melt outputs if they exist
      MeltOutputs<dim> *melt_out = out.template get_additional_output<MeltOutputs<dim> >();

      if (melt_out != NULL)
        {
          const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");

          for (unsigned int i=0; i<in.position.size(); ++i)
            {
              double porosity = std::max(in.composition[i][porosity_idx],0.0);

              melt_out->fluid_viscosities[i] = eta_f;
              melt_out->permeabilities[i] = reference_permeability * std::pow(porosity,3) * std::pow(1.0-porosity,2);
              melt_out->fluid_density_gradients[i] = Tensor<1,dim>();

              // temperature dependence of density is 1 - alpha * (T - T(adiabatic))
              double temperature_dependence = 1.0;
              if (this->include_adiabatic_heating ())
                temperature_dependence -= (in.temperature[i] - this->get_adiabatic_conditions().temperature(in.position[i]))
                                          * thermal_expansivity;
              else
                temperature_dependence -= (in.temperature[i] - reference_T) * thermal_expansivity;
              melt_out->fluid_densities[i] = melt_density * temperature_dependence
                                             * std::exp(melt_compressibility * (in.pressure[i] - this->get_surface_pressure()));

              melt_out->compaction_viscosities[i] = xi_0 * exp(- alpha_phi * porosity);

              double visc_temperature_dependence = 1.0;
              if (this->include_adiabatic_heating ())
                {
                  const double delta_temp = in.temperature[i]-this->get_adiabatic_conditions().temperature(in.position[i]);
                  visc_temperature_dependence = std::max(std::min(std::exp(-thermal_bulk_viscosity_exponent*delta_temp/this->get_adiabatic_conditions().temperature(in.position[i])),1e4),1e-4);
                }
              else
                {
                  const double delta_temp = in.temperature[i]-reference_T;
                  visc_temperature_dependence = std::max(std::min(std::exp(-thermal_bulk_viscosity_exponent*delta_temp/reference_T),1e4),1e-4);
                }
              melt_out->compaction_viscosities[i] *= visc_temperature_dependence;
            }
        }
    }



    template <int dim>
    void
    BinarySolidSolution<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Binary solid solution");
        {
          prm.declare_entry ("Reference forsterite density", "3060",
                             Patterns::Double (0),
                             "Reference density of solid forsterite $\\rho_{s,fo}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference fayalite density", "4430",
                             Patterns::Double (0),
                             "Reference density of solid fayalite $\\rho_{s,fa}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference melt density", "3000",
                             Patterns::Double (0),
                             "Reference density of the melt/fluid$\\rho_{f,0}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference temperature", "293",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. The reference temperature is used "
                             "in both the density and viscosity formulas. Units: $K$.");
          prm.declare_entry ("Reference shear viscosity", "5e20",
                             Patterns::Double (0),
                             "The value of the constant viscosity $\\eta_0$ of the solid matrix. "
                             "This viscosity may be modified by both temperature and porosity "
                             "dependencies. Units: $Pa s$.");
          prm.declare_entry ("Reference bulk viscosity", "1e22",
                             Patterns::Double (0),
                             "The value of the constant bulk viscosity $\\xi_0$ of the solid matrix. "
                             "This viscosity may be modified by both temperature and porosity "
                             "dependencies. Units: $Pa s$.");
          prm.declare_entry ("Reference melt viscosity", "10",
                             Patterns::Double (0),
                             "The value of the constant melt viscosity $\\eta_f$. Units: $Pa s$.");
          prm.declare_entry ("Exponential melt weakening factor", "27",
                             Patterns::Double (0),
                             "The porosity dependence of the viscosity. Units: dimensionless.");
          prm.declare_entry ("Thermal viscosity exponent", "0.0",
                             Patterns::Double (0),
                             "The temperature dependence of the shear viscosity. Dimensionless exponent. "
                             "See the general documentation "
                             "of this model for a formula that states the dependence of the "
                             "viscosity on this factor, which is called $\\beta$ there.");
          prm.declare_entry ("Thermal bulk viscosity exponent", "0.0",
                             Patterns::Double (0),
                             "The temperature dependence of the bulk viscosity. Dimensionless exponent. "
                             "See the general documentation "
                             "of this model for a formula that states the dependence of the "
                             "viscosity on this factor, which is called $\\beta$ there.");
          prm.declare_entry ("Thermal conductivity", "4.7",
                             Patterns::Double (0),
                             "The value of the thermal conductivity $k$. "
                             "Units: $W/m/K$.");
          prm.declare_entry ("Reference specific heat", "1250",
                             Patterns::Double (0),
                             "The value of the specific heat $C_p$. "
                             "Units: $J/kg/K$.");
          prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");
          prm.declare_entry ("Reference permeability", "1e-8",
                             Patterns::Double(),
                             "Reference permeability of the solid host rock."
                             "Units: $m^2$.");
          prm.declare_entry ("Solid forsterite compressibility", "8e-12",
                             Patterns::Double (0),
                             "The value of the compressibility of solid forsterite. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Solid fayalite compressibility", "8e-12",
                             Patterns::Double (0),
                             "The value of the compressibility of solid fayalite. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Melt compressibility", "1.7e-11",
                             Patterns::Double (0),
                             "The value of the compressibility of the melt. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Include melting and freezing", "true",
                             Patterns::Bool (),
                             "Whether to include melting and freezing (according to a simplified "
                             "linear melting approximation in the model (if true), or not (if "
                             "false).");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    BinarySolidSolution<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Binary solid solution");
        {
          forsterite_density                = prm.get_double ("Reference forsterite density");
          fayalite_density                  = prm.get_double ("Reference fayalite density");
          melt_density                      = prm.get_double ("Reference melt density");
          reference_T                       = prm.get_double ("Reference temperature");
          eta_0                             = prm.get_double ("Reference shear viscosity");
          xi_0                              = prm.get_double ("Reference bulk viscosity");
          eta_f                             = prm.get_double ("Reference melt viscosity");
          reference_permeability            = prm.get_double ("Reference permeability");
          thermal_viscosity_exponent        = prm.get_double ("Thermal viscosity exponent");
          thermal_bulk_viscosity_exponent   = prm.get_double ("Thermal bulk viscosity exponent");
          thermal_conductivity              = prm.get_double ("Thermal conductivity");
          reference_specific_heat           = prm.get_double ("Reference specific heat");
          thermal_expansivity               = prm.get_double ("Thermal expansion coefficient");
          alpha_phi                         = prm.get_double ("Exponential melt weakening factor");
          forsterite_compressibility        = prm.get_double ("Solid forsterite compressibility");
          fayalite_compressibility          = prm.get_double ("Solid forsterite compressibility");
          melt_compressibility              = prm.get_double ("Melt compressibility");
          include_melting_and_freezing      = prm.get_bool ("Include melting and freezing");

          if (thermal_viscosity_exponent!=0.0 && reference_T == 0.0)
            AssertThrow(false, ExcMessage("Error: Material model Melt simple with Thermal viscosity exponent can not have reference_T=0."));

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    BinarySolidSolution<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      if (this->get_parameters().use_operator_splitting
          && out.template get_additional_output<ReactionRateOutputs<dim> >() == NULL)
        {
          const unsigned int n_points = out.viscosities.size();
          out.additional_outputs.push_back(
            std_cxx11::shared_ptr<MaterialModel::AdditionalMaterialOutputs<dim> >
            (new MaterialModel::ReactionRateOutputs<dim> (n_points, this->n_compositional_fields())));
        }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(BinarySolidSolution,
                                   "binary solid solution",
                                   "A material model that implements melting and freezing for an olivine "
                                   "solid-solution model with the ideal end-members Mg2SiO4 (forsterite) "
                                   "and Fe2SiO4 (fayalite), as described in Phipps Morgan, Jason. "
                                   "Thermodynamics of pressure release melting of a veined plum pudding "
                                   "mantle. Geochemistry, Geophysics, Geosystems 2.4 (2001).")
  }
}
