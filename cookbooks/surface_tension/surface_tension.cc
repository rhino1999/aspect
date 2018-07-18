/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

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
#include <aspect/initial_composition/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/global.h>
#include <aspect/melt.h>
#include <aspect/simulator.h>
#include <aspect/simulator/assemblers/interface.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <array>


namespace aspect
{

  namespace InitialComposition
  {
    /**
     * An initial conditions model for sinusoidal melt bands
     * that implements a background porosity plus a plave wave in the porosity
     * field with a certain amplitude.
     */
    template <int dim>
    class MeltBandsInitialCondition : public Interface<dim>,
      public ::aspect::SimulatorAccess<dim>
    {
      public:

        /**
         * Return the initial porosity as a function of position.
         */
        virtual
        double
        initial_composition (const Point<dim> &position, const unsigned int n_comp) const;

        static
        void
        declare_parameters (ParameterHandler &prm);

        virtual
        void
        parse_parameters (ParameterHandler &prm);

        /**
         * Initialization function.
         */
        void
        initialize ();

        double
        get_wave_amplitude () const;

        double
        get_wave_number () const;

        double
        get_initial_band_angle () const;

      private:
        double amplitude;
        double background_porosity;
        double wave_number;
        double initial_band_angle;
    };
  }



  /**
   * This is the material model that can be used to models magmatic shear
   * bands including the effect of surface tension as described in the
   * following paper:
   * @code
   *  @article{bercovici2016mechanism,
   *  title={A mechanism for mode selection in melt band instabilities},
   *  author={Bercovici, David and Rudge, John F},
   *  journal={Earth and Planetary Science Letters},
   *  volume={433},
   *  pages={139--145},
   *  year={2016},
   *  publisher={Elsevier}
   *  }
   * @endcode
   *
   */
  namespace MaterialModel
  {
    using namespace dealii;

    namespace
    {
      std::vector<std::string> make_surface_tension_outputs_names()
      {
        std::vector<std::string> names;
        names.emplace_back("interface_areas");
        names.emplace_back("interface_curvatures");
        names.emplace_back("interface_curvature_variations");
        names.emplace_back("surface_tensions");
        names.emplace_back("growth_rates");
        names.emplace_back("modelled_growth_rates");
        return names;
      }
    }

    /**
     * Additional output fields for the surface tension material parameters
     * to be added to the MaterialModel::MaterialModelOutputs structure
     * and filled in the MaterialModel::Interface::evaluate() function.
     */
    template <int dim>
    class SurfaceTensionOutputs : public NamedAdditionalMaterialOutputs<dim>
    {
      public:
        SurfaceTensionOutputs(const unsigned int n_points);

        virtual std::vector<double> get_nth_output(const unsigned int idx) const;

        /**
         * Microscopic (pore and grain) scale interface area A, which
         * is assumed to be only a function of porosity.
         */
        std::vector<double> interface_areas;

        /**
         * Interface area curvature, or the derivative of the interface
         * area function with respect to the porosity.
         */
        std::vector<double> interface_curvatures;

        /**
         * Derivative of the interface area curvature with respect to the
         * porosity, or the second derivative of the interface area function
         * with respect to the porosity.
         */
        std::vector<double> interface_curvature_variations;

        /**
         * Surface tension on the interface between phases.
         */
        std::vector<double> surface_tensions;

        /**
         * Growth rate of magmatic shear bands computed analytically.
         */
        std::vector<double> growth_rates;

        /**
         * Growth rate of magmatic shear bands computed numerically.
         */
        std::vector<double> modelled_growth_rates;
    };


    template <int dim>
    SurfaceTensionOutputs<dim>::SurfaceTensionOutputs (const unsigned int n_points)
      :
      NamedAdditionalMaterialOutputs<dim>(make_surface_tension_outputs_names()),
      interface_areas(n_points, numbers::signaling_nan<double>()),
      interface_curvatures(n_points, numbers::signaling_nan<double>()),
      interface_curvature_variations(n_points, numbers::signaling_nan<double>()),
      surface_tensions(n_points, numbers::signaling_nan<double>()),
      growth_rates(n_points, numbers::signaling_nan<double>()),
	  modelled_growth_rates(n_points, numbers::signaling_nan<double>())
    {}

    template <int dim>
    std::vector<double>
    SurfaceTensionOutputs<dim>::get_nth_output(const unsigned int idx) const
    {
      AssertIndexRange (idx, 6);
      switch (idx)
        {
          case 0:
            return interface_areas;

          case 1:
            return interface_curvatures;

          case 2:
            return interface_curvature_variations;

          case 3:
            return surface_tensions;

          case 4:
            return growth_rates;

          case 5:
            return modelled_growth_rates;

          default:
            AssertThrow(false, ExcInternalError());
        }
      // We will never get here, so just return something
      return interface_areas;
    }

    /**
     * @note This benchmark only talks about the flow field, not about a
     * temperature field. All quantities related to the temperature are
     * therefore set to zero in the implementation of this class.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class ShearBandsTensionMaterial : public MeltInterface<dim>,
      public ::aspect::SimulatorAccess<dim>
    {
      public:
        virtual bool is_compressible () const
        {
          return false;
        }

        virtual double reference_viscosity () const
        {
          return eta_0;
        }

        virtual double reference_darcy_coefficient () const
        {
          return reference_permeability * std::pow(background_porosity,permeability_exponent) * std::pow(grainsize,2) / eta_f;
        }


        double
        get_background_porosity () const;

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

        virtual
        void
        create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const;


        virtual void evaluate(const typename MaterialModel::Interface<dim>::MaterialModelInputs &in,
                              typename MaterialModel::Interface<dim>::MaterialModelOutputs &out) const
        {
          const unsigned int porosity_idx = this->introspection().compositional_index_for_name("porosity");
          const double strain_rate_dependence = (1.0 - strain_rate_exponent) / (2. * strain_rate_exponent);

          for (unsigned int i=0; i<in.position.size(); ++i)
            {
              const double porosity = std::max(in.composition[i][porosity_idx],0.0);
              out.viscosities[i] = eta_0 * std::exp(-porosity_exponent*(porosity - background_porosity));

              if (in.strain_rate.size())
                {
                  const SymmetricTensor<2,dim> shear_strain_rate = in.strain_rate[i]
                                                                   - 1./dim * trace(in.strain_rate[i]) * unit_symmetric_tensor<dim>();
                  const double second_strain_rate_invariant = std::sqrt(std::abs(second_invariant(shear_strain_rate)));

                  if (std::abs(second_strain_rate_invariant) > 1e-30)
                    out.viscosities[i] *= std::pow(std::pow(second_strain_rate_invariant,2)
                                                   /std::pow(reference_strain_rate_invariant,2), strain_rate_dependence);
                }

              out.densities[i] = reference_rho_s;
              out.thermal_expansion_coefficients[i] = 0.0;
              out.specific_heat[i] = 1.0;
              out.thermal_conductivities[i] = 0.0;
              out.compressibilities[i] = 0.0;
              for (unsigned int c=0; c<in.composition[i].size(); ++c)
                out.reaction_terms[i][c] = 0.0;
            }

          // fill melt outputs if they exist
          MeltOutputs<dim> *melt_out = out.template get_additional_output<MeltOutputs<dim> >();

          if (melt_out != NULL)
            for (unsigned int i=0; i<in.position.size(); ++i)
              {
                double porosity = std::max(in.composition[i][porosity_idx],0.0);

                melt_out->compaction_viscosities[i] = std::pow(1.-porosity,2) * 5./3. * eta_0;
                melt_out->fluid_viscosities[i]= eta_f;
                melt_out->permeabilities[i]= reference_permeability * std::pow(porosity,permeability_exponent) * std::pow(grainsize,2);
                melt_out->fluid_densities[i]= reference_rho_f;
                melt_out->fluid_density_gradients[i] = 0.0;

                porosity = std::max(in.composition[i][porosity_idx],1e-8);
                melt_out->compaction_viscosities[i] = 10. * eta_0 * pow(porosity/background_porosity,-1.0);
              }

          // fill tension outputs if they exist
          SurfaceTensionOutputs<dim> *tension_out = out.template get_additional_output<SurfaceTensionOutputs<dim> >();
          if (tension_out != NULL)
          {
              // Prepare the field function
        	  const QGauss<dim> quadrature_formula (this->get_fe().base_element(this->introspection().base_elements.velocities).degree+1);
        	  Functions::FEFieldFunction<dim, DoFHandler<dim>, LinearAlgebra::BlockVector>
        	  fe_values(this->get_dof_handler(), this->get_solution(), this->get_mapping());

        	  std::vector<double> velocity_divergences(in.position.size());

              for(unsigned int d=0; d<1; ++d)
              {
              fe_values.set_active_cell(in.current_cell);
              std::vector<Tensor<1,dim> > velocity_gradients(in.position.size());
              fe_values.gradient_list(in.position,
            		               velocity_gradients,
								   this->introspection().component_indices.velocities[d]);
              for (unsigned int i=0; i<in.position.size(); ++i)
                velocity_divergences[i]+= velocity_gradients[i][d*(1+dim)];
              }

            for (unsigned int i=0; i<in.position.size(); ++i)
              {
                const double porosity = std::max(in.composition[i][porosity_idx],0.0);

                const double area_prefactor = geometry_factor * surface_tension / grainsize;
                tension_out->interface_areas[i] = area_prefactor * (1 + porosity_area_factor * std::pow(porosity,0.5));
                tension_out->interface_curvatures[i] = 0.5 * area_prefactor * porosity_area_factor * std::pow(porosity,-0.5);
                tension_out->interface_curvature_variations[i]= -0.25 * area_prefactor * porosity_area_factor * std::pow(porosity,-1.5);
                tension_out->surface_tensions[i] = surface_tension;

                // set up all the constants we need to compute the growth rate
                const double c_0 = eta_f / (reference_permeability * std::pow(grainsize,2)) * std::pow(porosity,2.-permeability_exponent);
                const double B_0 = std::pow(1.-background_porosity,2) * 5./3. * eta_0;
                const double compaction_length = sqrt(std::pow(background_porosity,2) * (B_0 + 4./3. * eta_0) / c_0);

                double second_strain_rate_invariant = reference_strain_rate_invariant;
                if (in.strain_rate.size())
                  {
                    const SymmetricTensor<2,dim> shear_strain_rate = in.strain_rate[i]
                                                                     - 1./dim * trace(in.strain_rate[i]) * unit_symmetric_tensor<dim>();
                    if (std::abs(second_invariant(shear_strain_rate)) > 100. * std::numeric_limits<double>::epsilon())
                      second_strain_rate_invariant = std::sqrt(std::abs(second_invariant(shear_strain_rate)));
                  }

                const double nu = eta_0 / (B_0 + 4./3. * eta_0);
                const double Gamma = tension_out->surface_tensions[i] * (1.0 - background_porosity) * tension_out->interface_curvature_variations[i]
                                     / (2. * second_strain_rate_invariant * eta_0);
                const double D = 1./(tension_out->interface_curvature_variations[i] * tension_out->interface_areas[i] * pow(compaction_length,2));
                const double Q = porosity_exponent / Gamma;
                const double q = 1.0 - 1./strain_rate_exponent;

                const InitialComposition::MeltBandsInitialCondition<dim> &melt_bands
                  = this->get_initial_composition_manager().template
                    get_matching_initial_composition_model<InitialComposition::MeltBandsInitialCondition<dim> > ();

                const double wave_number = melt_bands.get_wave_number();
                const double band_angle  = melt_bands.get_initial_band_angle();

//                std::cout << "nu = " << nu
//                		  << ", D = " << D
//						  << ", Q = " << Q
//						  << ", q = " << q
//						  << ", Gamma = " << Gamma
//						  << ", wave_number = " << wave_number
//						  << ", sine 2 band_angle = " << sin(2*band_angle)
//						  << ", prefactor = " << (1.0 - background_porosity) * nu * Gamma * std::pow(wave_number,2)
//						  << ", numerator = " << (Q * sin(2*band_angle) - (1. + D*pow(wave_number,2)) * (1. - q * pow(cos(2*band_angle),2)))
//						  << ", D = " << ((1. + pow(wave_number,2)) * (1. - q * pow(cos(2*band_angle),2)) - q * nu * pow(wave_number,2) * pow(sin(2*band_angle),2))
//						  << std::endl;

                tension_out->growth_rates[i] = (1.0 - background_porosity) * nu * Gamma * std::pow(wave_number,2)
                                               * (Q * sin(2*band_angle) - (1. + D*pow(wave_number,2)) * (1. - q * pow(cos(2*band_angle),2)))
                                               / ((1. + pow(wave_number,2)) * (1. - q * pow(cos(2*band_angle),2)) - q * nu * pow(wave_number,2) * pow(sin(2*band_angle),2));

                // we multiply by 2 epsilon_0 to get back to real units
                tension_out->growth_rates[i] *= (2. * reference_strain_rate_invariant);

                if(in.strain_rate.size())
                {

                  tension_out->modelled_growth_rates[i] = (1.0 - background_porosity) / (melt_bands.get_wave_amplitude() * background_porosity)
				                                          * velocity_divergences[i];
                }

                tension_out->surface_tensions[i] = 0.0;
              }
          }
        }

      private:
        double reference_rho_s;
        double reference_rho_f;
        double eta_0;
        double eta_f;
        double reference_permeability;
        double strain_rate_exponent;
        double porosity_exponent;
        double background_porosity;
        double permeability_exponent;
        double reference_strain_rate_invariant;
        double grainsize;

        // surface tension parameters
        double geometry_factor;
        double porosity_area_factor;
        double surface_tension;
    };

    template <int dim>
    double
    ShearBandsTensionMaterial<dim>::get_background_porosity () const
    {
      return background_porosity;
    }


    template <int dim>
    void
    ShearBandsTensionMaterial<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Shear bands tension material");
        {
          prm.declare_entry ("Reference solid density", "3000",
                             Patterns::Double (0),
                             "Reference density of the solid $\\rho_{s,0}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference melt density", "3000",
                             Patterns::Double (0),
                             "Reference density of the melt/fluid$\\rho_{f,0}$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference shear viscosity", "6.e11",
                             Patterns::Double (0),
                             "The value of the constant viscosity $\\eta_0$ of the solid matrix. "
                             "Units: $Pa s$.");
          prm.declare_entry ("Reference melt viscosity", "10.0",
                             Patterns::Double (0),
                             "The value of the constant melt viscosity $\\eta_f$. Units: $Pa s$.");
          prm.declare_entry ("Reference permeability", "6.25e-4",
                             Patterns::Double(0),
                             "Reference permeability of the solid host rock."
                             "Units: $m^2$.");
          prm.declare_entry ("Strain rate exponent", "1.0",
                             Patterns::Double(0),
                             "Power-law exponent $n_{dis}$ for dislocation creep. "
                             "Units: none.");
          prm.declare_entry ("Porosity exponent", "25.0",
                             Patterns::Double(),
                             "Exponent $b$ for the exponential porosity-dependence "
                             "of the viscosity. "
                             "Units: none.");
          prm.declare_entry ("Background porosity", "0.03",
                             Patterns::Double (0),
                             "Background porosity used in the viscosity law. Units: none.");
          prm.declare_entry ("Permeability exponent", "2.0",
                             Patterns::Double(0),
                             "Power-law exponent $n$ for the porosity-dependence of permeability. "
                             "Units: none.");
          prm.declare_entry ("Reference strain rate", "3.e-4",
                             Patterns::Double(0),
                             "The reference strain rate used to compute the shear viscosity. "
                             "Units: 1/s.");
          prm.declare_entry ("Grain size", "2.e-6",
                             Patterns::Double(0),
                             "The grain size used to compute the permeability and the surface area. "
                             "Units: m.");
          prm.declare_entry ("Geometry factor", "3.34",
                             Patterns::Double(0),
                             "Geometry factor that is used to compute the the surface area of "
                             "grain-grain contact. "
                             "Units: none.");
          prm.declare_entry ("Porosity area factor", "-0.35",
                             Patterns::Double(),
                             "The prefactor of the porosity in the expression that is used to compute "
                             "the surface area of grain-grain contact. "
                             "Units: none.");
          prm.declare_entry ("Surface tension", "1.0",
                             Patterns::Double(0),
                             "The surface tension . "
                             "Units: Pa m.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

    }



    template <int dim>
    void
    ShearBandsTensionMaterial<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Shear bands tension material");
        {
          reference_rho_s            = prm.get_double ("Reference solid density");
          reference_rho_f            = prm.get_double ("Reference melt density");
          eta_0                      = prm.get_double ("Reference shear viscosity");
          eta_f                      = prm.get_double ("Reference melt viscosity");
          reference_permeability     = prm.get_double ("Reference permeability");
          strain_rate_exponent       = prm.get_double ("Strain rate exponent");
          porosity_exponent          = prm.get_double ("Porosity exponent");
          background_porosity        = prm.get_double ("Background porosity");
          permeability_exponent      = prm.get_double ("Permeability exponent");
          reference_strain_rate_invariant = prm.get_double ("Reference strain rate");
          grainsize                  = prm.get_double ("Grain size");
          geometry_factor            = prm.get_double ("Geometry factor");
          porosity_area_factor       = prm.get_double ("Porosity area factor");
          surface_tension            = prm.get_double ("Surface tension");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Declare dependencies on solution variables
      this->model_dependence.viscosity = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::strain_rate | NonlinearDependence::compositional_fields;
      this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
    }

    template <int dim>
    void
    ShearBandsTensionMaterial<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      if (out.template get_additional_output<SurfaceTensionOutputs<dim> >() == NULL)
        {
          const unsigned int n_points = out.viscosities.size();
          out.additional_outputs.push_back(
            std::shared_ptr<SurfaceTensionOutputs<dim> >
            (new SurfaceTensionOutputs<dim> (n_points)));
        }
    }
  }

  namespace InitialComposition
  {
    template <int dim>
    void
    MeltBandsInitialCondition<dim>::initialize ()
    {
      if (dynamic_cast<const MaterialModel::ShearBandsTensionMaterial<dim> *>(&this->get_material_model()) != NULL)
        {
          const MaterialModel::ShearBandsTensionMaterial<dim> *
          material_model
            = dynamic_cast<const MaterialModel::ShearBandsTensionMaterial<dim> *>(&this->get_material_model());

          background_porosity = material_model->get_background_porosity();
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("Initial condition plane wave melt bands only "
                                 "works with the material model shear bands tension material."));
        }


      AssertThrow(amplitude < 1.0,
                  ExcMessage("Amplitude of the melt bands must be smaller "
                             "than the background porosity."));
    }


    template <int dim>
    double
	MeltBandsInitialCondition<dim>::get_wave_amplitude () const
    {
      return amplitude;
    }


    template <int dim>
    double
    MeltBandsInitialCondition<dim>::get_wave_number () const
    {
      return wave_number;
    }


    template <int dim>
    double
    MeltBandsInitialCondition<dim>::get_initial_band_angle () const
    {
      return initial_band_angle;
    }


    template <int dim>
    double
    MeltBandsInitialCondition<dim>::
    initial_composition (const Point<dim> &position, const unsigned int /*n_comp*/) const
    {
      return background_porosity * (1.0 + amplitude * cos(wave_number*position[0]*sin(initial_band_angle)
                                                          + wave_number*position[1]*cos(initial_band_angle)));
    }


    template <int dim>
    void
    MeltBandsInitialCondition<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Plane wave melt bands initial condition");
        {
          prm.declare_entry ("Wave amplitude", "1e-4",
                             Patterns::Double (0),
                             "Amplitude of the plane wave added to the initial "
                             "porosity. Units: none.");
          prm.declare_entry ("Wave number", "2000",
                             Patterns::Double (0),
                             "Wave number of the plane wave added to the initial "
                             "porosity. Is multiplied by 2 pi internally. "
                             "Units: 1/m.");
          prm.declare_entry ("Initial band angle", "30",
                             Patterns::Double (0),
                             "Initial angle of the plane wave added to the initial "
                             "porosity. Units: degrees.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    MeltBandsInitialCondition<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Plane wave melt bands initial condition");
        {
          amplitude          = prm.get_double ("Wave amplitude");
          wave_number        = 2.0 * numbers::PI * prm.get_double ("Wave number");
          initial_band_angle = 2.0 * numbers::PI / 360.0 * prm.get_double ("Initial band angle");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }

  namespace Assemblers
  {
    namespace
    {
      template <int dim>
      bool
      is_velocity_or_pressures (const Introspection<dim> &introspection,
                                const unsigned int p_c_component_index,
                                const unsigned int p_f_component_index,
                                const unsigned int component_index)
      {
        if (component_index == p_c_component_index)
          return true;

        if (component_index == p_f_component_index)
          return true;

        for (unsigned int i=0; i<dim; ++i)
          if (component_index == introspection.component_indices.velocities[i])
            return true;

        return false;
      }
    }

    /**
     * Surface tension assembler.
     */
    template <int dim>
    class SurfaceTensionAssembler :
      public Interface<dim>,
      public SimulatorAccess<dim>
    {

      public:

        virtual
        void
        execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                 internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
        {
          internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
          internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

          const Introspection<dim> &introspection = this->introspection();
          const FiniteElement<dim> &fe            = this->get_fe();
          const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
          const unsigned int n_q_points           = scratch.finite_element_values.n_quadrature_points;

          // We need the spatial derivatives of the porosity to compute the surface tension
          std::vector<Tensor<1,dim> > porosity_gradients(n_q_points, numbers::signaling_nan<Tensor<1,dim>>());
          std::vector<double> porosity_laplacians(n_q_points, numbers::signaling_nan<double>());

          const unsigned int porosity_index = introspection.compositional_index_for_name("porosity");
          scratch.finite_element_values[introspection.extractors.compositional_fields[porosity_index]].get_function_gradients (
            this->get_solution(), porosity_gradients);
          scratch.finite_element_values[introspection.extractors.compositional_fields[porosity_index]].get_function_laplacians (
            this->get_solution(), porosity_laplacians);

          const unsigned int p_f_component_index = introspection.variable("fluid pressure").first_component_index;
          const unsigned int p_c_component_index = introspection.variable("compaction pressure").first_component_index;

          const MaterialModel::SurfaceTensionOutputs<dim>
          *tension = scratch.material_model_outputs.template get_additional_output<MaterialModel::SurfaceTensionOutputs<dim> >();

          Assert(tension != NULL,
                 ExcMessage("Error: The surface tension terms can only be assembled if the material model creates "
                            "and fills the surface tension outputs!"));


          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              const unsigned int component_index_i = fe.system_to_component_index(i).first;

              if (is_velocity_or_pressures(introspection,p_c_component_index,p_f_component_index,component_index_i))
                {
                  data.local_dof_indices[i_stokes] = scratch.local_dof_indices[i];
                  ++i_stokes;
                }
              ++i;
            }

          for (unsigned int q=0; q<n_q_points; ++q)
            {
              for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
                {
                  const unsigned int component_index_i = fe.system_to_component_index(i).first;

                  if (is_velocity_or_pressures(introspection,p_c_component_index,p_f_component_index,component_index_i))
                    {
                      scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);

                      ++i_stokes;
                    }
                  ++i;
                }

              const double porosity = std::max(scratch.material_model_inputs.composition[q][porosity_index],0.0);

              const double JxW = scratch.finite_element_values.JxW(q);

              for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
                {
                  data.local_rhs(i) += (tension->surface_tensions[q] * (1.0 - porosity)
                                        *
                                        // term that describes the effective pressure gradient due                      to
                                        // variations in microscopic interface curvature
                                        ((- tension->interface_curvatures[q] * scratch.div_phi_u[i])
                                         +
                                         // term that describes pressure gradients caused by variations
                                         // in the surface tension on effective macroscopic (diffuse)
                                         // interfaces associated with sharp gradients in the porosity
                                         (1.0 / tension->interface_areas[q]
                                          * porosity_laplacians[q] * scratch.div_phi_u[i]))
                                       )
                                       * JxW;
                }
            }
        }


        virtual
        void
        create_additional_material_model_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
        {
          if (out.template get_additional_output<MaterialModel::SurfaceTensionOutputs<dim> >() == NULL)
            {
              const unsigned int n_points = out.viscosities.size();
              out.additional_outputs.push_back(
                std::shared_ptr<MaterialModel::SurfaceTensionOutputs<dim> >
                (new MaterialModel::SurfaceTensionOutputs<dim> (n_points)));
            }
        }
    };


    /**
     * Surface tension face assembler.
     */
    template <int dim>
    class SurfaceTensionFaceAssembler :
      public Interface<dim>,
      public SimulatorAccess<dim>
    {

      public:
        virtual
        void
        execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                 internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
        {
          internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
          internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

          const Introspection<dim> &introspection = this->introspection();
          const FiniteElement<dim> &fe            = this->get_fe();
          const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
          const unsigned int face_n_q_points = scratch.face_finite_element_values.n_quadrature_points;

          // We need the spatial derivatives of the porosity to compute the surface tension
          std::vector<double> porosity_laplacians(face_n_q_points, numbers::signaling_nan<double>());
          const unsigned int porosity_index = introspection.compositional_index_for_name("porosity");

          const unsigned int p_f_component_index = introspection.variable("fluid pressure").first_component_index;
          const unsigned int p_c_component_index = introspection.variable("compaction pressure").first_component_index;

          const MaterialModel::SurfaceTensionOutputs<dim>
          *face_tension = scratch.face_material_model_outputs.template get_additional_output<MaterialModel::SurfaceTensionOutputs<dim> >();

          Assert(face_tension != NULL,
                 ExcMessage("Error: The surface tension terms can only be assembled if the material model creates "
                            "and fills the surface tension outputs!"));

          scratch.face_finite_element_values.reinit (scratch.cell, scratch.face_number);
          scratch.face_finite_element_values[introspection.extractors.compositional_fields[porosity_index]].get_function_laplacians (
            this->get_solution(), porosity_laplacians);

          for (unsigned int q=0; q<face_n_q_points; ++q)
            {
              const double porosity = std::max(scratch.face_material_model_inputs.composition[q][porosity_index],0.0);

              const Tensor<1,dim> normal_vector = scratch.face_finite_element_values.normal_vector(q);
              const double JxW = scratch.face_finite_element_values.JxW(q);

              for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
                {
                  const unsigned int component_index_i = fe.system_to_component_index(i).first;

                  if (is_velocity_or_pressures(introspection,p_c_component_index,p_f_component_index,component_index_i))
                    {
                      // apply the fluid pressure boundary condition
                      data.local_rhs(i_stokes) += face_tension->surface_tensions[q] * (1.0 - porosity)
                                                  *
                                                  (face_tension->interface_curvatures[q]
                                                   - porosity_laplacians[q] / face_tension->interface_areas[q])
                                                  * scratch.face_finite_element_values[introspection.extractors.velocities].value(i,q)
                                                  * normal_vector
                                                  * JxW;
                      ++i_stokes;
                    }
                  ++i;
                }
            }
        }


        virtual
        void
        create_additional_material_model_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
        {
          if (out.template get_additional_output<MaterialModel::SurfaceTensionOutputs<dim> >() == NULL)
            {
              const unsigned int n_points = out.viscosities.size();
              out.additional_outputs.push_back(
                std::shared_ptr<MaterialModel::SurfaceTensionOutputs<dim> >
                (new MaterialModel::SurfaceTensionOutputs<dim> (n_points)));
            }
        }
    };

    template <int dim>
    void set_assemblers_surface_tension(const SimulatorAccess<dim> &simulator_access,
                                        Assemblers::Manager<dim> &assemblers)
    {
      AssertThrow (dynamic_cast<const MaterialModel::ShearBandsTensionMaterial<dim>*>
                   (&simulator_access.get_material_model()) != 0,
                   ExcMessage ("The surface tension assembler can only be used with the "
                               "material model 'shear bands tension material'!"));

      SurfaceTensionAssembler<dim> *surface_tension_assembler = new SurfaceTensionAssembler<dim>();
      assemblers.stokes_system.push_back (std::unique_ptr<SurfaceTensionAssembler<dim> > (surface_tension_assembler));
      assemblers.stokes_system_assembler_properties.needed_update_flags = update_hessians
                                                                          |
                                                                          assemblers.stokes_system_assembler_properties.needed_update_flags;

      SurfaceTensionFaceAssembler<dim> *surface_tension_face_assembler = new SurfaceTensionFaceAssembler<dim>();
      assemblers.stokes_system_on_boundary_face.push_back (std::unique_ptr<SurfaceTensionFaceAssembler<dim> > (surface_tension_face_assembler));
      assemblers.stokes_system_assembler_on_boundary_face_properties.needed_update_flags = update_hessians
          |
          assemblers.stokes_system_assembler_on_boundary_face_properties.needed_update_flags;
    }
  }
}

template <int dim>
void signal_connector (aspect::SimulatorSignals<dim> &signals)
{
  signals.set_assemblers.connect (&aspect::Assemblers::set_assemblers_surface_tension<dim>);
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(ShearBandsTensionMaterial,
                                   "shear bands tension material",
                                   "A material model that corresponds to the setup to"
                                   "generate magmatic shear bands described in Bercovici et al., "
                                   "EPSL, 2016.")
  }

  namespace InitialComposition
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(MeltBandsInitialCondition,
                                              "melt bands initial condition",
                                              "Composition is set to background porosity plus "
                                              "plane wave.")
  }
}

ASPECT_REGISTER_SIGNALS_CONNECTOR(signal_connector<2>,
                                  signal_connector<3>)
