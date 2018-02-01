/*
  Copyright (C) 2015 - 2018 by the authors of the ASPECT code.

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

#include <aspect/material_model/interface.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/boundary_composition/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/melt.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/parsed_function.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    template <int dim>
    class InnerCoreMelt : public MaterialModel::MeltInterface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Return whether the model is compressible or not.  Incompressibility
         * does not necessarily imply that the density is constant; rather, it
         * may still depend on temperature or pressure. In the current
         * context, compressibility means whether we should solve the continuity
         * equation as $\nabla \cdot (\rho \mathbf u)=0$ (compressible Stokes)
         * or as $\nabla \cdot \mathbf{u}=0$ (incompressible Stokes).
         */
        virtual bool is_compressible () const;

        virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                              MaterialModel::MaterialModelOutputs<dim> &out) const;

        /**
         * @name Reference quantities
         * @{
         */
        virtual double reference_viscosity () const;

        virtual double reference_darcy_coefficient () const;

        /**
         * Declare the parameters this class takes through input files. The
         * default implementation of this function does not describe any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         * The default implementation of this function does not read any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
        double reference_rho_s;
        double reference_rho_f;
        double reference_T;
        double eta;
        double eta_f;
        double thermal_expansivity;
        double specific_heat;
        double thermal_conductivity;
        double reference_permeability;
    };

  }
}

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    double
    InnerCoreMelt<dim>::
    reference_viscosity () const
    {
      return eta;
    }

    template <int dim>
    double
    InnerCoreMelt<dim>::
    reference_darcy_coefficient () const
    {
      const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
      const Point<dim> surface_point = this->get_geometry_model().representative_point(0.0);
      const double surface_porosity = this->get_boundary_composition_manager().boundary_composition(0, surface_point, porosity_idx);

      // 0.01 = 1% melt
      const double phi_relative = 0.01 / surface_porosity;
      return reference_permeability * std::pow(phi_relative,3) / eta_f;
    }

    template <int dim>
    bool
    InnerCoreMelt<dim>::
    is_compressible () const
    {
      return false;
    }

    template <int dim>
    void
    InnerCoreMelt<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      for (unsigned int i=0; i < in.position.size(); ++i)
        {
          out.viscosities[i] = eta;

          // We linearize the temperature dependence of density, as temperature variations are small.
          // As this is only the solid density, we do not include the dependence on composition (porosity) here.
          out.densities[i] = reference_rho_s * (1 - thermal_expansivity * (in.temperature[i] - reference_T));

          out.thermal_expansion_coefficients[i] = thermal_expansivity;
          out.specific_heat[i] = specific_heat;
          out.thermal_conductivities[i] = thermal_conductivity;

          // the model is incompressible, as the density stays almost constant
          out.compressibilities[i] = 0.0;

          // Pressure derivative of entropy at the given positions.
          out.entropy_derivative_pressure[i] = 0.0;
          // Temperature derivative of entropy at the given positions.
          out.entropy_derivative_temperature[i] = 0.0;

          // Change in composition due to chemical reactions at the
          // given positions. The term reaction_terms[i][c] is the
          // change in compositional field c at point i.
          // For now, there is no melting or freezing.
          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;
        }

      // Adding compaction: from "A model for sedimentary compaction of a viscous medium and its
      // application to inner-core growth"

      // boundary conditions:
      // ps = pf at outer boundary
      // growth rate of the column = solid velocity + sedimentation rate
      // surface porosity phi_0 = 0.5

      // constitutive relationships
      // Blake-Kozeny-Carmann equation for permeability
      // note that we are solving for (1 - phi) * DeltaP compared to their equation (16),
      // and therefore we have to divide their effective bulk viscosity by 1 - phi to get our compaction viscosity
      // no compaction for phi > phi_0

      // fill melt outputs if they exist
      MeltOutputs<dim> *melt_out = out.template get_additional_output<MeltOutputs<dim> >();

      if (melt_out != NULL)
        {
          const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");

          for (unsigned int i=0; i<in.position.size(); ++i)
            {
              double porosity = std::max(in.composition[i][porosity_idx],0.0);

              melt_out->fluid_viscosities[i] = eta_f;

              // Blake-Kozeny-Carmann equation for permeability, defined relative to a reference porosity
              const Point<dim> surface_point = this->get_geometry_model().representative_point(0.0);
              const double surface_porosity = this->get_boundary_composition_manager().boundary_composition(0, surface_point, porosity_idx);

              melt_out->permeabilities[i] = reference_permeability * std::pow(porosity,3) / (1 - std::pow(porosity,2))
                                            * (1 - std::pow(surface_porosity,2)) / std::pow(surface_porosity,3);

              melt_out->fluid_density_gradients[i] = Tensor<1,dim>();

              // temperature dependence of density is 1 - alpha * (T - T_ref)
              double temperature_dependence = 1.0 - (in.temperature[i] - reference_T) * thermal_expansivity;
              melt_out->fluid_densities[i] = reference_rho_f * temperature_dependence;

              // Note that we are solving for p_c = (1 - phi) * DeltaP compared to their equation (16),
              // and therefore we have to divide their effective bulk viscosity by (1 - phi) to get our compaction viscosity
              porosity = std::max(in.composition[i][porosity_idx],1e-4);
              melt_out->compaction_viscosities[i] = 4.0/3.0 * out.viscosities[i] / porosity;
            }
        }
    }


    template <int dim>
    void
    InnerCoreMelt<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Inner core melt");
        {
          prm.declare_entry ("Reference solid density", "12800",
                             Patterns::Double (0),
                             "Reference density of the solid $\\rho_{s,0}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference melt density", "12200",
                             Patterns::Double (0),
                             "Reference density of the melt/fluid$\\rho_{f,0}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference temperature", "5600",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. The reference temperature is used "
                             "in both the density and viscosity formulas. Units: K.");
          prm.declare_entry ("Shear viscosity", "8e21",
                             Patterns::Double (0),
                             "The value of the constant viscosity $\\eta_0$ of the solid matrix. "
                             "This viscosity may be modified by both temperature and porosity "
                             "dependencies. Units: Pa s.");
          prm.declare_entry ("Melt viscosity", "1.22e-3",
                             Patterns::Double (0),
                             "The value of the constant melt viscosity $\\eta_f$. Units: Pa s.");
          prm.declare_entry ("Thermal conductivity", "100.0",
                             Patterns::Double (0),
                             "The value of the thermal conductivity $k$. "
                             "Units: W/m/K.");
          prm.declare_entry ("Specific heat", "800.0",
                             Patterns::Double (0),
                             "The value of the specific heat $C_p$. "
                             "Units: J/kg/K.");
          prm.declare_entry ("Thermal expansion coefficient", "1.1e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: 1/K.");
          prm.declare_entry ("Reference permeability", "2.7e-17",
                             Patterns::Double(),
                             "Reference permeability of the solid host rock."
                             "Units: $m^2$.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    InnerCoreMelt<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Inner core melt");
        {
          reference_rho_s                   = prm.get_double ("Reference solid density");
          reference_rho_f                   = prm.get_double ("Reference melt density");
          reference_T                       = prm.get_double ("Reference temperature");
          eta                               = prm.get_double ("Shear viscosity");
          eta_f                             = prm.get_double ("Melt viscosity");
          reference_permeability            = prm.get_double ("Reference permeability");
          thermal_conductivity              = prm.get_double ("Thermal conductivity");
          specific_heat                     = prm.get_double ("Specific heat");
          thermal_expansivity               = prm.get_double ("Thermal expansion coefficient");
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
    ASPECT_REGISTER_MATERIAL_MODEL(InnerCoreMelt,
                                   "inner core melt material",
                                   "A simple material model for compaction of the "
                                   "inner core.")
  }
}
