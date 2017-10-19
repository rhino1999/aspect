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
      //BC = 0.89;% fraction between
      const double P = pressure; // pressure is assumed to be in GPa
      const double T = temperature;   // tmperature in K
      const double R = constants::gas_constant;       // Ideal Gas Constant

      const double T_Fo_melting_surface = 2163.0; // Kelvin at 1 atmosphere - reference melting temperature for forsterite
      const double T_Fa_melting_surface = 1478.0; // Kelvin at 1 atmosphere - reference melting temperature for forsterite

      const double dS_Fo = 60.0; // entropy change in J/mol K
      const double dS_Fa = 60.0; // entropy change in J/mol K

      const double vliq_ref = 49.0/std::pow(100.0,3); // m3/mol
      const double vsol_ref_Fo = 46.0/std::pow(100.0,3); // m3/mol
      const double vsol_ref_Fa = 46.0/std::pow(100.0,3); // m3/mol

      const double surface_pressure = 1.e5; // 1 atmosphere = 1e5 Pa

      const double beta_sol_Fo = 8.e-3/1.e9; //1/Pa
      const double beta_sol_Fa = 8.e-3/1.e9; //1/Pa
      const double beta_liq = 1.7e-2/1.e9; //1/Pa

      // Melting Temperatures of Pure end-members as a function of Pressure
      // Eq. (A13) of Phipps Morgan 2001
      const double T_Fo_melting = T_Fo_melting_surface
                                  + (((vliq_ref - vsol_ref_Fo) * (P-surface_pressure))
                                      - ((vliq_ref * beta_liq - vsol_ref_Fo * beta_sol_Fo)
                                          * (std::pow(P,2) - std::pow(surface_pressure,2))))/dS_Fo;
      const double T_Fa_melting = T_Fa_melting_surface
                                  + (((vliq_ref - vsol_ref_Fa) * (P-surface_pressure))
                                      - ((vliq_ref * beta_liq - vsol_ref_Fa * beta_sol_Fa)
                                          * (std::pow(P,2) - std::pow(surface_pressure,2))))/dS_Fa;

      // Free Energy Change due to Melting
      const double dG_Fo_T_P = (T_Fo_melting_surface - T) * dS_Fo
                               + (vliq_ref - vsol_ref_Fo) * (P-surface_pressure)
                               - ((vliq_ref * beta_liq - vsol_ref_Fo * beta_sol_Fo) * (std::pow(P,2) - std::pow(surface_pressure,2)));
      const double dG_Fa_T_P = (T_Fa_melting_surface - T) * dS_Fa
                               + (vliq_ref - vsol_ref_Fa) * (P-surface_pressure)
                               - ((vliq_ref * beta_liq - vsol_ref_Fa * beta_sol_Fa) * (std::pow(P,2) - std::pow(surface_pressure,2)));

      // Mole Fraction of Each Component in Coexisting Liquids and Solids
      // const double Xl_Fa = (1.0 - exp(dG_Fo_T_P/(2.0*R*T)))/(exp(dG_Fa_T_P/(2.0*R*T))-exp(dG_Fo_T_P/(2.0*R*T)));
      // const double Xs_Fa = Xl_Fa * exp(dG_Fa_T_P/(2.0*R*T));
      double Xl_Fo = 1.0 - (1.0 - exp(dG_Fo_T_P/(2.0*R*T)))/(exp(dG_Fa_T_P/(2.0*R*T))-exp(dG_Fo_T_P/(2.0*R*T)));
      double Xs_Fo = Xl_Fo * exp(dG_Fo_T_P/(2.0*R*T));

      double melt_fraction = 0.0;
      if (T >= T_Fo_melting)
        melt_fraction = 1.0;
      else if (T <= T_Fa_melting)
        melt_fraction = 0.0;
      else
        melt_fraction = (bulk_composition - Xs_Fo)/(Xl_Fo - Xs_Fo);

      if (melt_fraction <= 0)
        {
          melt_fraction = 0;
          new_solid_composition = bulk_composition;
        }
      else if (melt_fraction >= 1)
        {
          melt_fraction = 1;
          new_melt_composition = bulk_composition;
        }
      else
        {
          new_solid_composition = Xs_Fo;
          new_melt_composition = Xl_Fo;
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
      std::vector<double> old_porosity(in.position.size());
      std::vector<double> old_solid_composition(in.position.size());
      std::vector<double> old_melt_composition(in.position.size());

      ReactionRateOutputs<dim> *reaction_rate_out = out.template get_additional_output<ReactionRateOutputs<dim> >();

      // make sure the compositional fields we want to use exist
      if (this->include_melt_transport())
        AssertThrow(this->introspection().compositional_name_exists("porosity"),
                    ExcMessage("Material model Melt simple with melt transport only "
                               "works if there is a compositional field called porosity."));

      if (this->include_melt_transport() && include_melting_and_freezing)
        AssertThrow(this->introspection().compositional_name_exists("solid_composition"),
                    ExcMessage("Material model Melt global with melting and freezing "
                               "only works if there is a compositional field called "
                               "solid_composition."));

      if (this->include_melt_transport() && include_melting_and_freezing)
        AssertThrow(this->introspection().compositional_name_exists("melt_composition"),
                    ExcMessage("Material model Melt global with melting and freezing "
                               "only works if there is a compositional field called "
                               "melt_composition."));

      // we want to get the porosity field from the old solution here,
      // because we need a field that is not updated in the nonlinear iterations
      if (this->include_melt_transport() && in.current_cell.state() == IteratorState::valid
          && this->get_timestep_number() > 0 && !this->get_parameters().use_operator_splitting
          && include_melting_and_freezing)
        {
          // Prepare the field function
          Functions::FEFieldFunction<dim, DoFHandler<dim>, LinearAlgebra::BlockVector>
          fe_value(this->get_dof_handler(), this->get_old_solution(), this->get_mapping());

          const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
          const unsigned int solid_composition_idx = this->introspection().compositional_index_for_name("solid_composition");
          const unsigned int melt_composition_idx = this->introspection().compositional_index_for_name("melt_composition");

          fe_value.set_active_cell(in.current_cell);
          fe_value.value_list(in.position,
                              old_porosity,
                              this->introspection().component_indices.compositional_fields[porosity_idx]);
          fe_value.value_list(in.position,
                              old_solid_composition,
                              this->introspection().component_indices.compositional_fields[solid_composition_idx]);
          fe_value.value_list(in.position,
                              old_melt_composition,
                              this->introspection().component_indices.compositional_fields[melt_composition_idx]);
        }
      else if (this->get_parameters().use_operator_splitting && this->include_melt_transport() && include_melting_and_freezing)
        for (unsigned int i=0; i<in.position.size(); ++i)
          {
            const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
            old_porosity[i] = in.composition[i][porosity_idx];
            old_solid_composition[i] = in.composition[i][this->introspection().compositional_index_for_name("solid_composition")];
            old_melt_composition[i] = in.composition[i][this->introspection().compositional_index_for_name("melt_composition")];
          }

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
          const double delta_rho = this->introspection().compositional_name_exists("solid_composition")
                                   ?
                                   depletion_density_change * in.composition[i][this->introspection().compositional_index_for_name("solid_composition")]
                                   :
                                   0.0;
          out.densities[i] = (reference_rho_s + delta_rho) * temperature_dependence
                             * std::exp(compressibility * (in.pressure[i] - this->get_surface_pressure()));

          // now compute melting and crystallization
          if (this->include_melt_transport() && include_melting_and_freezing)
            {
              const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
              const unsigned int solid_composition_idx = this->introspection().compositional_index_for_name("solid_composition");
              const unsigned int melt_composition_idx = this->introspection().compositional_index_for_name("melt_composition");

              const double porosity = std::min(1.0, std::max(in.composition[i][porosity_idx],0.0));
              const double solid_composition = std::min(1.0, std::max(old_solid_composition[i],0.0));
              const double melt_composition = std::min(1.0, std::max(old_melt_composition[i],0.0));

              const double bulk_composition = porosity * melt_composition + (1.0 - porosity) * solid_composition;

              // calculate the melting rate as difference between the equilibrium melt fraction
              // and the solution of the previous time step
              double new_solid_composition = solid_composition;
              double new_melt_composition = melt_composition;
              const double eq_melt_fraction = melt_fraction(in.temperature[i],
                                                            this->get_adiabatic_conditions().pressure(in.position[i]),
                                                            bulk_composition,
                                                            new_solid_composition,
                                                            new_melt_composition);

              // do not allow negative porosity or porosity > 1
              const double melting_rate = limit_update_to_0_and_1(old_porosity[i], eq_melt_fraction - old_porosity[i]);
              const double change_of_solid_composition = limit_update_to_0_and_1(old_solid_composition[i], new_solid_composition - old_solid_composition[i]);
              const double change_of_melt_composition = limit_update_to_0_and_1(old_melt_composition[i], new_melt_composition - old_melt_composition[i]);

              for (unsigned int c=0; c<in.composition[i].size(); ++c)
                {
//                  if (c == solid_composition_idx && this->get_timestep_number() > 1 && (in.strain_rate.size()))
//                    out.reaction_terms[i][c] = change_of_solid_composition
//                                               - in.composition[i][c] * trace(in.strain_rate[i]) * this->get_timestep();
//                  else if (c == melt_composition_idx && this->get_timestep_number() > 1 && (in.strain_rate.size()))
//                    out.reaction_terms[i][c] = change_of_melt_composition;
//                  else if (c == porosity_idx && this->get_timestep_number() > 1 && (in.strain_rate.size()))
//                    out.reaction_terms[i][c] = melting_rate
//                                               * out.densities[i]  / this->get_timestep();
//                  else
//                    out.reaction_terms[i][c] = 0.0;

                  // fill reaction rate outputs if the model uses operator splitting
                  if (this->get_parameters().use_operator_splitting)
                    {
                      if (reaction_rate_out != NULL)
                        {
                          if (c == solid_composition_idx && this->get_timestep_number() > 0)
                            reaction_rate_out->reaction_rates[i][c] = change_of_solid_composition / this->get_timestep()
                                                                      - in.composition[i][c] * trace(in.strain_rate[i]);
                          else if (c == melt_composition_idx && this->get_timestep_number() > 0)
                            reaction_rate_out->reaction_rates[i][c] = change_of_melt_composition / this->get_timestep();
                          else if (c == porosity_idx && this->get_timestep_number() > 0)
                            reaction_rate_out->reaction_rates[i][c] = melting_rate / this->get_timestep();
                          else
                            reaction_rate_out->reaction_rates[i][c] = 0.0;
                        }
                      out.reaction_terms[i][c] = 0.0;
                    }
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
              melt_out->fluid_densities[i] = reference_rho_f * temperature_dependence
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
          prm.declare_entry ("Reference solid density", "3000",
                             Patterns::Double (0),
                             "Reference density of the solid $\\rho_{s,0}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference melt density", "2500",
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
          prm.declare_entry ("Depletion density change", "0.0",
                             Patterns::Double (),
                             "The density contrast between material with a depletion of 1 and a "
                             "depletion of zero. Negative values indicate lower densities of "
                             "depleted material. Depletion is indicated by the compositional "
                             "field with the name peridotite. Not used if this field does not "
                             "exist in the model. "
                             "Units: $kg/m^3$.");
          prm.declare_entry ("Surface solidus", "1300",
                             Patterns::Double (0),
                             "Solidus for a pressure of zero. "
                             "Units: $K$.");
          prm.declare_entry ("Depletion solidus change", "200.0",
                             Patterns::Double (),
                             "The solidus temperature change for a depletion of 100\\%. For positive "
                             "values, the solidus gets increased for a positive peridotite field "
                             "(depletion) and lowered for a negative peridotite field (enrichment). "
                             "Scaling with depletion is linear. Only active when fractional melting "
                             "is used. "
                             "Units: $K$.");
          prm.declare_entry ("Pressure solidus change", "6e-8",
                             Patterns::Double (),
                             "The linear solidus temperature change with pressure. For positive "
                             "values, the solidus gets increased for positive pressures. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Solid compressibility", "0.0",
                             Patterns::Double (0),
                             "The value of the compressibility of the solid matrix. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Melt compressibility", "0.0",
                             Patterns::Double (0),
                             "The value of the compressibility of the melt. "
                             "Units: $1/Pa$.");
          prm.declare_entry ("Melt bulk modulus derivative", "0.0",
                             Patterns::Double (0),
                             "The value of the pressure derivative of the melt bulk "
                             "modulus. "
                             "Units: None.");
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
          reference_rho_s                   = prm.get_double ("Reference solid density");
          reference_rho_f                   = prm.get_double ("Reference melt density");
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
          depletion_density_change          = prm.get_double ("Depletion density change");
          surface_solidus                   = prm.get_double ("Surface solidus");
          depletion_solidus_change          = prm.get_double ("Depletion solidus change");
          pressure_solidus_change           = prm.get_double ("Pressure solidus change");
          compressibility                   = prm.get_double ("Solid compressibility");
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
                                   "A material model that implements a simple formulation of the "
                                   "material parameters required for the modelling of melt transport, "
                                   "including a source term for the porosity according to a simplified "
                                   "linear melting model similar to \\cite{schmeling2006}:\n"
                                   "$\\phi_\\text{equilibrium} = \\frac{T-T_\\text{sol}}{T_\\text{liq}-T_\\text{sol}}$\n"
                                   "with "
                                   "$T_\\text{sol} = T_\\text{sol,0} + \\Delta T_p \\, p + \\Delta T_c \\, C$ \n"
                                   "$T_\\text{liq} = T_\\text{sol}  + \\Delta T_\\text{sol-liq}$.")
  }
}
