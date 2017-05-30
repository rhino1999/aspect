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

#ifndef _aspect_material_model_forsterite_fayalite_h
#define _aspect_material_model_forsterite_fayalite_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    template <int dim>
    class RockComponent: public SimulatorAccess<dim>
    {
      public:
      RockComponent(ParameterHandler &prm);

      virtual ~RockComponent();

      virtual void initialize();

      /**
       * Return the chemical potential of the given endmember.
       */
      double get_chemical_potential(const double temperature,
                                    const double pressure) const;

      /**
       * Return the indefinite integral of specific heat integrated
       * with respect to temperature $\\int Cp_s(T) dt$.
       */
      double get_integrated_specific_heat(const double temperature) const;

      /**
       * Return the indefinite integral of specific heat divided by
       * temperature integrated with respect to temperature $\\int Cp_s(T)/T dt$.
       */
      double get_integrated_specific_heat_over_T(const double temperature) const;

      /**
       * Return molar fraction of a component, given its mass fraction as an
       * input.
       */
      double convert_mass_to_mol_fractions(const double mass_fraction,
                                           const PhaseState phase) const;

      /**
       * Return molar fraction of a component, given its mass fraction as an
       * input.
       */
      double convert_mol_to_mass_fractions(const double molar_fraction,
                                           const PhaseState phase) const;

      private:
      /**
       * Coefficients used in the polynomial that describes the
       * specific heat capacity (in dependence of temperature and pressure).
       */
      std::vector<double> specific_heat_coefficients;

      /**
       * Coefficients for the equation of state parameters for the solid volume
       * $V_s$,
       * $\\frac{\\partial V_s}{\\partial T}$,
       * $\\frac{\\partial V_s}{\\partial p}$,
       * $\\frac{\\partial^2 V_s}{\\partial T \\partial p}$,
       * $\\frac{\\partial^2 V_s}{\\partial p^2}$
       */
      std::vector<double> volume_coefficients_solid;

      /**
       * Coefficients for the equation of state parameters for the liquid volume
       * $V_l$,
       * $\\frac{\\partial V_l}{\\partial T}$,
       * $\\frac{\\partial V_l}{\\partial p}$,
       * $\\frac{\\partial^2 V_l}{\\partial T \\partial p}$,
       * $\\frac{\\partial^2 V_l}{\\partial p^2}$
       */
      std::vector<double> volume_coefficients_liquid;

      double molar_mass_solid;
      double molar_mass_liquid;

      /**
       * Entropy and enthalpy of this component for the reference temperature
       * and reference pressure.
       */
      double reference_enthalpy;
      double reference_entropy;

      double reference_temperature;
      double reference_pressure;
    };

    /**
     * A material model that uses the parameterizations from the Berman
     * model for forsterite and fayalite to compute material properties.
     *
     * The model is considered incompressible, following the definition
     * described in Interface::is_compressible. This is essentially the
     * material model used in the step-32 tutorial program.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class ForsteriteFayalite : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                              MaterialModel::MaterialModelOutputs<dim> &out) const;

        /**
         * @name Qualitative properties one can ask a material model
         * @{
         */

        /**
         * Return whether the model is compressible or not.  Incompressibility
         * does not necessarily imply that the density is constant; rather, it
         * may still depend on temperature or pressure. In the current
         * context, compressibility means whether we should solve the continuity
         * equation as $\nabla \cdot (\rho \mathbf u)=0$ (compressible Stokes)
         * or as $\nabla \cdot \mathbf{u}=0$ (incompressible Stokes).
         */
        virtual bool is_compressible () const;
        /**
         * @}
         */

        /**
         * @name Reference quantities
         * @{
         */
        virtual double reference_viscosity () const;
        /**
         * @}
         */

        /**
         * @name Functions used in dealing with run-time parameters
         * @{
         */
        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);
        /**
         * @}
         */

      private:
        double
        compute_chemical_potentials(const double temperature,
                                    const double pressure,
                                    const double composition,
                                    const RockComponent component) const;

        std::vector<double>
        compute_reaction_rates(const double temperature,
                               const double pressure,
                               const double porosity,
                               const double solid_composition,
                               const double liquid_composition) const;

        RockComponent forsterite;
        RockComponent fayalite;

        /**
         * An enum indicating whether we want to compute quantities for
         * the solid or the liquid phase.
         */
        enum PhaseState {solid, liquid};

        double reference_rho;
        double reference_T;
        double eta;
        double composition_viscosity_prefactor;
        double thermal_viscosity_exponent;
        double maximum_thermal_prefactor;
        double minimum_thermal_prefactor;
        double thermal_alpha;
        double reference_specific_heat;

        /**
         * The thermal conductivity.
         */
        double k_value;

        double compositional_delta_rho;


    };

  }
}

#endif
