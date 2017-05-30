/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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


#include <aspect/material_model/forsterite_fayalite.h>

using namespace dealii;

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    double
    RockComponent<dim>::
    get_integrated_specific_heat(const double temperature) const
    {
      // indefinite integral  of $\int Cps(T)dt$
      return specific_heat_coefficients[0]*temperature
             + 2.0 * specific_heat_coefficients[1] * sqrt(temperature)
             - 1./temperature*(specific_heat_coefficients[2] + 0.5 * specific_heat_coefficients[3] * 1./temperature);
    }


    template <int dim>
    double
    RockComponent<dim>::
    get_integrated_specific_heat_over_T(const double temperature) const
    {
      // indefinite integral  of $\int Cps(T)dt$
      return specific_heat_coefficients[0] * log(temperature)
             - 2.0 * specific_heat_coefficients[1]/sqrt(temperature)
             - pow(1./temperature,2) * (0.5*specific_heat_coefficients[2] + specific_heat_coefficients[3] * 1./(3.0 * temperature));
    }


    template <int dim>
    double
    RockComponent<dim>::
    get_chemical_potential(const double temperature,
                           const double pressure) const
    {
      intCps = get_integrated_specific_heat(temperature) - get_integrated_specific_heat(reference_temperature);
      intCpsonT = get_integrated_specific_heat_over_T(temperature) - get_integrated_specific_heat_over_T(reference_temperature);

      // definite integral /int_Pr^P dVsdT dP
      const double dT = temperature - reference_temperature;
      const double dP = pressure - reference_pressure;
      const double intdVdT = dP * (self.v[3] + 2.0 * self.v[4] * dT);

      mu = reference_enthalpy - temperature * reference_entropy +  intCps - temperature * intCpsonT + intdVdT;
      return mu;
    }


    template <int dim>
    double
    RockComponent<dim>::
    convert_mass_to_mol_fractions(const double mass_fraction,
                                  const PhaseState phase) const
    {
      const double molar_mass = (phase == solid
                                 ?
                                 molar_mass_solid
                                 :
                                 molar_mass_liquid);
      return mass_fraction * molar_mass;
    }


    template <int dim>
    double
    RockComponent<dim>::
    convert_mol_to_mass_fractions(const double molar_fraction,
                                  const PhaseState phase) const
    {
      const double molar_mass = (phase == solid
                                 ?
                                 molar_mass_solid
                                 :
                                 molar_mass_liquid);
      return molar_fraction / molar_mass;
    }


    template <int dim>
    double
    ForsteriteFayalite<dim>::
    compute_chemical_potentials(const double temperature,
                                const double pressure,
                                const double composition,
                                const RockComponent component) const
    {
      // TODO: we do not need this function, this can be computed as a one-liner where we need it
      return component.get_chemical_potential(temperature, pressure)
             + constants::gas_constant * temperature * log(composition);
    }


    template <int dim>
    std::vector<double>
    ForsteriteFayalite<dim>::
    compute_reaction_rates(const double temperature,
                           const double pressure,
                           const double porosity,
                           const double solid_composition,
                           const double liquid_composition) const
    {
      std::vector<double> gamma;



      // Availability function:  checks whether a component is available,
      // returns 0 if either the phase or concentration of a reactant is 0
      const unsigned int avaialability_solid = (solid_composition*(1.-porosity)>0 ? 1 : 0);
      const unsigned int avaialability_liquid = (liquid_composition*porosity>0 ? 1 : 0);

      // compute reaction term
      // def R(phi,c,P,T):

      // Finally, the actual Reaction terms
      // def gamma(self,P,T,C):
      // return array of reactions in mass units
       // Each reaction is of form  Gamma_j = r_j*rho_j*A_j/RT
       // where r_j are rate constants, rho_j is the density of the product and
       // A_j/RT is the scaled affinity

      // covert mass concentration to mol fraction
      const double mol_fraction = component.convert_mass_to_mol_fractions(C);
      gamma = self.rates * self.rho_j(temperature, pressure) * self.A(temperature, pressure, mol_fraction) / (constants::gas_constant * temperature);

      gamma = Gamma.gamma(P,T,C(c));
      // loop over points if they exist (we'll use this later for PDE's)

      for (unsigned int i=0; i<compositional_fields.size(); ++i)
        {
          // if Gamma is positive, check that solid reactants are available
          if (gamma[i] > 0.)
            gamma[j] *= avaialability_solid;
          // if Gamma is negative, check that fluid components are available
          else
            gamma[j] *= avaialability_liquid;
        }

      return gamma;
    }


    template <int dim>
    void
    ForsteriteFayalite<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");

      for (unsigned int i=0; i < in.position.size(); ++i)
        {
          const double delta_temp = in.temperature[i]-reference_T;
          const double temperature_dependence = (reference_T > 0
                                                 ?
                                                 std::max(std::min(std::exp(-thermal_viscosity_exponent*delta_temp/reference_T),
                                                                   maximum_thermal_prefactor),
                                                          minimum_thermal_prefactor)
                                                 :
                                                 1.0);

          out.viscosities[i] = ((composition_viscosity_prefactor != 1.0) && (in.composition[i].size()>0))
                               ?
                               //Geometric interpolation
                               pow(10.0, ((1-in.composition[i][0]) * log10(eta*temperature_dependence)
                                          + in.composition[i][0] * log10(eta*composition_viscosity_prefactor*temperature_dependence)))
                               :
                               temperature_dependence * eta;

          const double c = (in.composition[i].size()>0)
                           ?
                           std::max(0.0, in.composition[i][0])
                           :
                           0.0;

          out.densities[i] = reference_rho * (1 - thermal_alpha * (in.temperature[i] - reference_T))
                             + compositional_delta_rho * c;

          out.thermal_expansion_coefficients[i] = thermal_alpha;

          // specific heat
          cs = in.composition[i][0];
          out.specific_heat[i] = cs[0] + cs[1] / sqrt(in.temperature[i]) + pow(1./in.temperature[i],2)*(cs[2] + cs[3]*1./in.temperature[i]);

          out.thermal_conductivities[i] = k_value;
          out.compressibilities[i] = 0.0;
          // Pressure derivative of entropy at the given positions.
          out.entropy_derivative_pressure[i] = 0.0;
          // Temperature derivative of entropy at the given positions.
          out.entropy_derivative_temperature[i] = 0.0;
          // Change in composition due to chemical reactions at the
          // given positions. The term reaction_terms[i][c] is the
          // change in compositional field c at point i.
          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;
        }

      // fill melt reaction rates if they exist
      ReactionRateOutputs<dim> *reaction_out = out.template get_additional_output<ReactionRateOutputs<dim> >();

      if (reaction_out != NULL)
        {
          for (unsigned int q=0; q < in.position.size(); ++q)
            {
              // dx/dt = alpha * x + beta * x * y
              reaction_out->reaction_rates[0][q] = alpha / time_scale * in.composition[q][0]
                                                   - beta / time_scale * in.composition[q][0] * in.composition[q][1];

              // dy/dt = gamma * y + delta * x * y
              reaction_out->reaction_rates[1][q] = - gamma / time_scale * in.composition[q][1]
                                                   +  delta / time_scale * in.composition[q][0] * in.composition[q][1];
            }
        }
    }


    template <int dim>
    double
    ForsteriteFayalite<dim>::
    reference_viscosity () const
    {
      return eta;
    }



    template <int dim>
    bool
    ForsteriteFayalite<dim>::
    is_compressible () const
    {
      return false;
    }



    template <int dim>
    void
    ForsteriteFayalite<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Forsterite fayalite model");
        {
          prm.declare_entry ("Reference density", "3300",
                             Patterns::Double (0),
                             "Reference density $\\rho_0$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference temperature", "293",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. The reference temperature is used "
                             "in both the density and viscosity formulas. Units: $K$.");
          prm.declare_entry ("Viscosity", "5e24",
                             Patterns::Double (0),
                             "The value of the constant viscosity $\\eta_0$. This viscosity may be "
                             "modified by both temperature and compositional dependencies. Units: $kg/m/s$.");
          prm.declare_entry ("Composition viscosity prefactor", "1.0",
                             Patterns::Double (0),
                             "A linear dependency of viscosity on the first compositional field. "
                             "Dimensionless prefactor. With a value of 1.0 (the default) the "
                             "viscosity does not depend on the composition. See the general documentation "
                             "of this model for a formula that states the dependence of the "
                             "viscosity on this factor, which is called $\\xi$ there.");
          prm.declare_entry ("Thermal viscosity exponent", "0.0",
                             Patterns::Double (0),
                             "The temperature dependence of viscosity. Dimensionless exponent. "
                             "See the general documentation "
                             "of this model for a formula that states the dependence of the "
                             "viscosity on this factor, which is called $\\beta$ there.");
          prm.declare_entry("Maximum thermal prefactor","1.0e2",
                            Patterns::Double (0),
                            "The maximum value of the viscosity prefactor associated with temperature "
                            "dependence.");
          prm.declare_entry("Minimum thermal prefactor","1.0e-2",
                            Patterns::Double (0),
                            "The minimum value of the viscosity prefactor associated with temperature "
                            "dependence.");
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
                             "The value of the thermal expansion coefficient $\\alpha$. "
                             "Units: $1/K$.");
          prm.declare_entry ("Density differential for compositional field 1", "0",
                             Patterns::Double(),
                             "If compositional fields are used, then one would frequently want "
                             "to make the density depend on these fields. In this simple material "
                             "model, we make the following assumptions: if no compositional fields "
                             "are used in the current simulation, then the density is simply the usual "
                             "one with its linear dependence on the temperature. If there are compositional "
                             "fields, then the density only depends on the first one in such a way that "
                             "the density has an additional term of the kind $+\\Delta \\rho \\; c_1(\\mathbf x)$. "
                             "This parameter describes the value of $\\Delta \\rho$. Units: $kg/m^3/\\textrm{unit "
                             "change in composition}$.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    ForsteriteFayalite<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Forsterite fayalite model");
        {
          reference_rho              = prm.get_double ("Reference density");
          reference_T                = prm.get_double ("Reference temperature");
          eta                        = prm.get_double ("Viscosity");
          composition_viscosity_prefactor = prm.get_double ("Composition viscosity prefactor");
          thermal_viscosity_exponent = prm.get_double ("Thermal viscosity exponent");
          maximum_thermal_prefactor       = prm.get_double ("Maximum thermal prefactor");
          minimum_thermal_prefactor       = prm.get_double ("Minimum thermal prefactor");
          if ( maximum_thermal_prefactor == 0.0 ) maximum_thermal_prefactor = std::numeric_limits<double>::max();
          if ( minimum_thermal_prefactor == 0.0 ) minimum_thermal_prefactor = std::numeric_limits<double>::min();

          k_value                    = prm.get_double ("Thermal conductivity");
          reference_specific_heat    = prm.get_double ("Reference specific heat");
          thermal_alpha              = prm.get_double ("Thermal expansion coefficient");
          compositional_delta_rho    = prm.get_double ("Density differential for compositional field 1");

          if (thermal_viscosity_exponent!=0.0 && reference_T == 0.0)
            AssertThrow(false, ExcMessage("Error: Material model simple with Thermal viscosity exponent can not have reference_T=0."));
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Declare dependencies on solution variables
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
      this->model_dependence.viscosity = NonlinearDependence::none;
      this->model_dependence.density = NonlinearDependence::none;

      if (thermal_viscosity_exponent != 0)
        this->model_dependence.viscosity |= NonlinearDependence::temperature;
      if (composition_viscosity_prefactor != 1.0)
        this->model_dependence.viscosity |= NonlinearDependence::compositional_fields;

      if (thermal_alpha != 0)
        this->model_dependence.density |=NonlinearDependence::temperature;
      if (compositional_delta_rho != 0)
        this->model_dependence.density |=NonlinearDependence::compositional_fields;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(ForsteriteFayalite,
                                   "forsterite fayalite",
                                   "A material model that has constant values "
                                   "for all coefficients but the density and viscosity. The defaults for all "
                                   "coefficients are chosen to be similar to what is believed to be correct "
                                   "for Earth's mantle. All of the values that define this model are read "
                                   "from a section ``Material model/Forsterite fayalite model'' in the input file, see "
                                   "Section~\\ref{parameters:Material_20model/Forsterite fayalite_20model}."
                                   "\n\n"
                                   "This model uses the following set of equations for the two coefficients that "
                                   "are non-constant: "
                                   "\\begin{align}"
                                   "  \\eta(p,T,\\mathfrak c) &= \\tau(T) \\zeta(\\mathfrak c) \\eta_0, \\\\"
                                   "  \\rho(p,T,\\mathfrak c) &= \\left(1-\\alpha (T-T_0)\\right)\\rho_0 + \\Delta\\rho \\; c_0,"
                                   "\\end{align}"
                                   "where $c_0$ is the first component of the compositional vector "
                                   "$\\mathfrak c$ if the model uses compositional fields, or zero otherwise. "
                                   "\n\n"
                                   "The temperature pre-factor for the viscosity formula above is "
                                   "defined as "
                                   "\\begin{align}"
                                   "  \\tau(T) &= H\\left(e^{-\\beta (T-T_0)/T_0}\\right),"
                                   "  \\qquad\\qquad H(x) = \\begin{cases}"
                                   "                            \\tau_{min} & \\text{if}\\; x<\\tau_{min}, \\\\"
                                   "                            x & \\text{if}\\; 10^{-2}\\le x \\le 10^2, \\\\"
                                   "                            \\tau_{max} & \\text{if}\\; x>\\tau_{max}, \\\\"
                                   "                         \\end{cases}"
                                   "\\end{align} "
                                   "where $\\beta$ corresponds to the input parameter ``Thermal viscosity exponent'' "
                                   "and $T_0$ to the parameter ``Reference temperature''. If you set $T_0=0$ "
                                   "in the input file, the thermal pre-factor $\\tau(T)=1$. The parameters $\\tau_{min}$ "
                                   "and $\\tau_{max}$ set the minimum and maximum values of the temperature pre-factor "
                                   "and are set using ``Maximum thermal prefactor'' and ``Minimum thermal prefactor''. "
                                   "Specifying a value of 0.0 for the minimum or maximum values will disable pre-factor limiting."
                                   "\n\n"
                                   "The compositional pre-factor for the viscosity is defined as "
                                   "\\begin{align}"
                                   "  \\zeta(\\mathfrak c) &= \\xi^{c_0}"
                                   "\\end{align} "
                                   "if the model has compositional fields and equals one otherwise. $\\xi$ "
                                   "corresponds to the parameter ``Composition viscosity prefactor'' in the "
                                   "input file."
                                   "\n\n"
                                   "Finally, in the formula for the density, $\\alpha$ corresponds to the "
                                   "``Thermal expansion coefficient'' and "
                                   "$\\Delta\\rho$ "
                                   "corresponds to the parameter ``Density differential for compositional field 1''."
                                   "\n\n"
                                   "Note that this model uses the formulation that assumes an incompressible "
                                   "medium despite the fact that the density follows the law "
                                   "$\\rho(T)=\\rho_0(1-\\alpha(T-T_{\\text{ref}}))$. "
                                   "\n\n"
                                   "\\note{Despite its name, this material model is not exactly ``simple'', "
                                   "as indicated by the formulas above. While it was originally intended "
                                   "to be simple, it has over time acquired all sorts of temperature "
                                   "and compositional dependencies that weren't initially intended. "
                                   "Consequently, there is now a ``simpler'' material model that now fills "
                                   "the role the current model was originally intended to fill.}")
  }
}
