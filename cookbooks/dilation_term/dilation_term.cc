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

#include <array>
#include <utility>
#include <limits>
#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/parsed_function.h>
#include <aspect/heating_model/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * This material model takes any other material model as a base model,
     * and adds additional material model outputs defining a source term
     * for the mass conservation equations that can be defined as a
     * function depending on position and time in the input file.
     *
     * The method is described in the following paper:
     * @code
     * @article{theissen2011coupled,
     *   title={Coupled mechanical and hydrothermal modeling of crustal accretion at intermediate to fast spreading ridges},
     *   author={Theissen-Krah, Sonja and Iyer, Karthik and R{\"u}pke, Lars H and Morgan, Jason Phipps},
     *   journal={Earth and Planetary Science Letters},
     *   volume={311},
     *   number={3-4},
     *   pages={275--286},
     *   year={2011},
     *   publisher={Elsevier}
     * }
     * @endcode
     *
     * @ingroup MaterialModels
     */

    template <int dim>
    class DilationTerm : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Initialize the model at the beginning of the run.
         */
        virtual
        void initialize();

        /**
         * Update the base model and injection function at the beginning of
         * each timestep.
         */
        virtual
        void update();

        /**
         * Function to compute the material properties in @p out given the
         * inputs in @p in.
         */
        virtual
        void
        evaluate (const typename Interface<dim>::MaterialModelInputs &in,
                  typename Interface<dim>::MaterialModelOutputs &out) const;
        
        /**
         * Method to declare parameters related to depth-dependent model
         */
        static void
        declare_parameters (ParameterHandler &prm);

        /**
         * Method to parse parameters related to depth-dependent model
         */
        virtual void
        parse_parameters (ParameterHandler &prm);

        /**
         * Method that indicates whether material is compressible. Depth dependent model is compressible
         * if and only if base model is compressible.
         */
        virtual bool is_compressible () const;

        /**
         * Method to calculate reference viscosity for the depth-dependent model. The reference
         * viscosity is determined by evaluating the depth-dependent part of the viscosity at
         * the mean depth of the model.
         */
        virtual double reference_viscosity () const;

        virtual
        void
        create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const;

      private:
        /**
         * Parsed function that specifies the amount of material that is injected
         * into the model.
         */
        Functions::ParsedFunction<dim> injection_function;

        /**
         * Pointer to the material model used as the base model.
         */
        std::shared_ptr<MaterialModel::Interface<dim> > base_model;
    };
  }
}


namespace aspect
{
  namespace HeatingModel
  {
    using namespace dealii;

    /**
     * A class that implements the heating related to the injection of 
     * melt into the model. It takes the amount of material added on the
     * right-hand side of the mass conservation equation and adds the
     * corresponding heating terms to the energy equation (considering
     * the latent heat of crystallization and the different temeprature
     * of the injected melt). 
     *
     * @ingroup HeatingModels
     */
    template <int dim>
    class LatentHeatInjection : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Compute the heating model outputs for this class.
         */
        virtual
        void
        evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
                  const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
                  HeatingModel::HeatingModelOutputs &heating_model_outputs) const;

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
        /**
         * Properties of injected material.
         */
        double latent_heat_of_crystallization;
        double temperature_of_injected_melt;
    };
  }
}


namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void
    DilationTerm<dim>::initialize()
    {
      base_model->initialize();
    }


    template <int dim>
    void
    DilationTerm<dim>::update()
    {
      base_model->update();

      // we get time passed as seconds (always) but may want
      // to reinterpret it in years
      if (this->convert_output_to_years())
        injection_function.set_time (this->get_time() / year_in_seconds);
      else
        injection_function.set_time (this->get_time());
    }

    template <int dim>
    void
    DilationTerm<dim>::evaluate(const typename Interface<dim>::MaterialModelInputs &in,
                                typename Interface<dim>::MaterialModelOutputs &out) const
    {
      // fill variable out with the results form the base material model
      base_model -> evaluate(in,out);

      MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>
      *force = out.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >();

      for (unsigned int i=0; i < in.position.size(); ++i)
        {
          if (force)
            {
              for (unsigned int d=0; d < dim; ++d)
                force->rhs_u[i][d] = 0;

              force->rhs_p[i] = injection_function.value(in.position[i]);
            }
        }
    }

    template <int dim>
    void
    DilationTerm<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Dilation term");
        {
          prm.declare_entry("Base model","simple",
                            Patterns::Selection(MaterialModel::get_valid_model_names_pattern<dim>()),
                            "The name of a material model that will be modified by an "
                            "averaging operation. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Material models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.enter_subsection("Injection function");
          {
            Functions::ParsedFunction<dim>::declare_parameters(prm,1);
            prm.declare_entry("Function expression","0.0");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    DilationTerm<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Dilation term");
        {
          Assert( prm.get("Base model") != "dilation term",
                  ExcMessage("You may not use ``dilation term'' as the base model for "
                             "itself.") );

          // create the base model and initialize its SimulatorAccess base
          // class; it will get a chance to read its parameters below after we
          // leave the current section
          base_model.reset(create_material_model<dim>(prm.get("Base model")));
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(base_model.get()))
            sim->initialize_simulator (this->get_simulator());

          prm.enter_subsection("Injection function");
          {
            try
              {
                injection_function.parse_parameters(prm);
              }
            catch (...)
              {
                std::cerr << "FunctionParser failed to parse\n"
                          << "\t Injection function\n"
                          << "with expression \n"
                          << "\t' " << prm.get("Function expression") << "'";
                throw;
              }
            prm.leave_subsection();
          }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      /* After parsing the parameters for averaging, it is essential to parse
      parameters related to the base model. */
      base_model->parse_parameters(prm);
      this->model_dependence = base_model->get_model_dependence();
    }

    template <int dim>
    bool
    DilationTerm<dim>::
    is_compressible () const
    {
      return base_model->is_compressible();
    }

    template <int dim>
    double
    DilationTerm<dim>::
    reference_viscosity() const
    {
      // if material is injected, the divergence of the velocity is not zero anymore
      return true;
    }

    template <int dim>
    void
    DilationTerm<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // Because we use the force outputs in the heating model, we always have to attach them, not only in the
      // places where the RHS of the Stokes system is computed.
      if (out.template get_additional_output<AdditionalMaterialOutputsStokesRHS<dim> >() == nullptr)
        {
          const unsigned int n_points = out.viscosities.size();
          out.additional_outputs.push_back(
            std::shared_ptr<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >
            (new MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> (n_points)));
        }
    }
  }
}


namespace aspect
{
  namespace HeatingModel
  {
    template <int dim>
    void
    LatentHeatInjection<dim>::
    evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
              const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
              HeatingModel::HeatingModelOutputs &heating_model_outputs) const
    {
      Assert(heating_model_outputs.heating_source_terms.size() == material_model_inputs.position.size(),
             ExcMessage ("Heating outputs need to have the same number of entries as the material model inputs."));

      const MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>
      *force = material_model_outputs.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >();

      for (unsigned int q=0; q<heating_model_outputs.heating_source_terms.size(); ++q)
        {
          heating_model_outputs.heating_source_terms[q] = 0.0;
          heating_model_outputs.lhs_latent_heat_terms[q] = 0.0;
          heating_model_outputs.rates_of_temperature_change[q] = 0.0;

          if(force != nullptr)
          heating_model_outputs.heating_source_terms[q] =
            force->rhs_p[q] * (latent_heat_of_crystallization +
                               (temperature_of_injected_melt - material_model_inputs.temperature[q])
                               * material_model_outputs.densities[q] * material_model_outputs.specific_heat[q]);
        }
    }

    template <int dim>
    void
    LatentHeatInjection<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Latent heat injection");
        {
          prm.declare_entry ("Latent heat of crystallization", "1.1e9",
                             Patterns::Double(0),
                             "The latent heat of crystallization that is released when material "
                             "is injected into the model. "
                             "Units: J/m$^3$.");
          prm.declare_entry ("Temperature of injected melt", "1600",
                             Patterns::Double(0),
                             "The temperature of the material injected into the model. "
                             "Units: K.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    LatentHeatInjection<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Latent heat injection");
        {
          latent_heat_of_crystallization = prm.get_double ("Latent heat of crystallization");
          temperature_of_injected_melt = prm.get_double ("Temperature of injected melt");
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
    ASPECT_REGISTER_MATERIAL_MODEL(DilationTerm,
                                   "dilation term",
                                   "This material model uses a ``Base model'' from which material properties are "
                                   "derived. It then adds source terms in the mass conservation equation "
                                   "that describe the addition of melt to the model. "
                                   "The terms are described in Theissen-Krah et al., 2011.")
  }


  namespace HeatingModel
  {
    ASPECT_REGISTER_HEATING_MODEL(LatentHeatInjection,
                                  "latent heat injection",
                                  "Latent heat release due to the injection of melt into the model. "
                                  "This heating model takes the source term added to the mass "
                                  "conservation equation and adds the corresponding source term to "
                                  "the temperature equation. This source term includes both the "
                                  "effect of latent heat release upon crystallization and the fact "
                                  "that injected material might have a diffeent temperature.")
  }
}
