/*
  Copyright (C) 2014 by the authors of the ASPECT code.

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


#ifndef __aspect__model_damage_rheology_h
#define __aspect__model_damage_rheology_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * A material model that consists of globally constant values for all
     * material parameters except that the density decays linearly with the
     * temperature and the viscosity, which depends on the temperature,
     * pressure, strain rate and grain size.
     *
     * The grain size evolves in time, dependent on strain rate, temperature,
     * creep regime, and phase transitions.
     *
     * The model is considered compressible.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class DamageRheology : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * @name Qualitative properties one can ask a material model
         * @{
         */

        /**
         * Return true if the viscosity() function returns something that may
         * depend on the variable identifies by the argument.
         */
        virtual bool
        viscosity_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
         * Return true if the density() function returns something that may
         * depend on the variable identifies by the argument.
         */
        virtual bool
        density_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
         * Return true if the compressibility() function returns something
         * that may depend on the variable identifies by the argument.
         *
         * This function must return false for all possible arguments if the
         * is_compressible() function returns false.
         */
        virtual bool
        compressibility_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
         * Return true if the specific_heat() function returns something that
         * may depend on the variable identifies by the argument.
         */
        virtual bool
        specific_heat_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
         * Return true if the thermal_conductivity() function returns
         * something that may depend on the variable identifies by the
         * argument.
         */
        virtual bool
        thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const;

        /**
         * Return whether the model is compressible or not.  Incompressibility
         * does not necessarily imply that the density is constant; rather, it
         * may still depend on temperature or pressure. In the current
         * context, compressibility means whether we should solve the contuity
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

        virtual double reference_density () const;

        virtual void evaluate(const typename Interface<dim>::MaterialModelInputs &in,
                              typename Interface<dim>::MaterialModelOutputs &out) const;
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

        /**
         * Returns the ratio of dislocation to diffusion viscosity. Useful
         * for postprocessing purposes to determine the regime of deformation
         * in the viscosity ratio postprocessor.
         */
        double
        viscosity_ratio (const double temperature,
                         const double pressure,
                         const std::vector<double> &composition,
                         const SymmetricTensor<2,dim> &strain_rate,
                         const Point<dim> &position) const;

      private:
        double reference_rho;
        double reference_T;
        double eta;
        double composition_viscosity_prefactor_1;
        double composition_viscosity_prefactor_2;
        double compositional_delta_rho;
        double thermal_alpha;
        double reference_specific_heat;

        /**
         * The thermal conductivity.
         */
        double k_value;

        // grain evolution parameters
        double gas_constant; // J/(K*mol)
        double grain_growth_activation_energy;
        double grain_growth_activation_volume;
        double grain_growth_rate_constant;
        double grain_growth_exponent;
        double reciprocal_required_strain;
        double recrystallized_grain_size;

        // for paleowattmeter
        bool use_paleowattmeter;
        double grain_boundary_energy;
        double boundary_area_change_work_fraction;
        double geometric_constant;

        // rheology parameters
        double dislocation_creep_exponent;
        double dislocation_activation_energy;
        double dislocation_activation_volume;
        double dislocation_creep_prefactor;
        double diffusion_creep_exponent;
        double diffusion_activation_energy;
        double diffusion_activation_volume;
        double diffusion_creep_prefactor;
        double diffusion_creep_grain_size_exponent;
        double max_temperature_dependence_of_eta;

        virtual double viscosity (const double                  temperature,
                                  const double                  pressure,
                                  const std::vector<double>    &compositional_fields,
                                  const SymmetricTensor<2,dim> &strain_rate,
                                  const Point<dim>             &position) const;

        virtual double diffusion_viscosity (const double      temperature,
                                            const double      pressure,
                                            const double      grain_size,
                                            const Point<dim> &position) const;

        virtual double dislocation_viscosity (const double      temperature,
                                              const double      pressure,
                                              const double      second_strain_rate_invariant,
                                              const Point<dim> &position) const;

        virtual double density (const double temperature,
                                const double pressure,
                                const std::vector<double> &compositional_fields,
                                const Point<dim> &position) const;

        /**
         * Rate of grain size growth (Ostwald ripening) or reduction
         * (due to phase transformations) in dependence on temperature
         * pressure, strain rate, mineral phase and creep regime.
         * We use the grain size evolution laws described in Solomatov
         * and Reese, 2008. Grain size variations in the Earth’s mantle
         * and the evolution of primordial chemical heterogeneities,
         * J. Geophys. Res., 113, B07408.
         */
        virtual
        double
        grain_size_growth_rate (const double                  temperature,
                                const double                  pressure,
                                const std::vector<double>    &compositional_fields,
                                const SymmetricTensor<2,dim> &strain_rate,
                                const Tensor<1,dim>          &velocity,
                                const Point<dim>             &position,
                                const unsigned int            phase_index) const;

        /**
         * Function that defines the phase transition interface
         * (0 above, 1 below the phase transition).This is done
         * individually for each transition and summed up in the end.
         */
        virtual
        double
        phase_function (const Point<dim> &position,
                        const double temperature,
                        const double pressure,
                        const int phase) const;

        // list of depth, width and Clapeyron slopes for the different phase
        // transitions and in which phase they occur
        std::vector<double> transition_depths;
        std::vector<double> transition_temperatures;
        std::vector<double> transition_slopes;
        std::vector<std::string> transition_phases;
    };

  }
}

#endif
