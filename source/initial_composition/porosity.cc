/*
  Copyright (C) 2017 by the authors of the ASPECT code.

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


#include <aspect/initial_composition/porosity.h>
#include <aspect/initial_temperature/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/material_model/interface.h>
#include <aspect/melt.h>


namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    double
    Porosity<dim>::
    initial_composition (const Point<dim> &position,
                         const unsigned int compositional_index) const
    {
      const MaterialModel::MeltFractionModel<dim> *material_model =
        dynamic_cast<const MaterialModel::MeltFractionModel<dim>* > (&this->get_material_model());
      AssertThrow(material_model != NULL,
                  ExcMessage("The used material model is not derived from the 'MeltFractionModel' class, "
                             "and therefore does not support computing equilibrium melt fractions. "
                             "This is incompatible with the `porosity' "
                             "initial composition plugin, which needs to compute these melt fractions."));

      bool is_melt_field = false;
      for (unsigned int c=0; c<names_of_melt_fields.size(); ++c)
        {
          AssertThrow(this->introspection().compositional_name_exists(names_of_melt_fields[c]),
                      ExcMessage("The initial composition plugin `porosity' "
                                 "did not find a compositional field called " + names_of_melt_fields[c] +
                                 " to initialize. Please add a compositional field with this name."));

          if (compositional_index == this->introspection().compositional_index_for_name(names_of_melt_fields[c]))
            is_melt_field = true;
        }

      if (is_melt_field)
        {
          MaterialModel::MaterialModelInputs<dim> in(1, this->n_compositional_fields());

          in.position[0] = position;
          in.temperature[0] = this->get_initial_temperature_manager().initial_temperature(position);
          in.pressure[0] = this->get_adiabatic_conditions().pressure(position);
          in.pressure_gradient[0] = 0.0;
          in.velocity[0] = 0.0;

          // Use the initial composition, except for the fields that are computed here,
          // to prevent infinite recursion
          for (unsigned int i = 0; i < this->n_compositional_fields(); ++i)
            if (std::find(names_of_melt_fields.begin(), names_of_melt_fields.end(), this->introspection().name_for_compositional_index(i))
                == names_of_melt_fields.end())
              in.composition[0][i] = this->get_initial_composition_manager().initial_composition(position,i);
            else
              in.composition[0][i] = 0.0;

          in.strain_rate[0] = SymmetricTensor<2,dim>();

          std::vector<double> melt_fraction(1);
          material_model->melt_fractions(in,melt_fraction);
          return melt_fraction[0];
        }
      return 0.0;
    }


    template <int dim>
    void
    Porosity<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Porosity");
        {
          prm.declare_entry ("Names of melt fields", "porosity",
                             Patterns::List(Patterns::Anything()),
                             "A user-defined name for each of the compositional fields that "
                             "the `porosity' initial condition should be applied to.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    Porosity<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("Porosity");
        {
          names_of_melt_fields = Utilities::split_string_list (prm.get("Names of melt fields"));

          // check that the names use only allowed characters, are not empty strings and are unique
          for (unsigned int c=0; c<names_of_melt_fields.size(); ++c)
            {
              Assert (names_of_melt_fields[c].find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                                                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                                         "0123456789_") == std::string::npos,
                      ExcMessage("Invalid character in field " + names_of_melt_fields[c] + ". "
                                 "Names of compositional fields should consist of a "
                                 "combination of letters, numbers and underscores."));
              Assert (names_of_melt_fields[c].size() > 0,
                      ExcMessage("Invalid name of field " + names_of_melt_fields[c] + ". "
                                 "Names of compositional fields need to be non-empty."));
              for (unsigned int j=0; j<c; ++j)
                Assert (names_of_melt_fields[c] != names_of_melt_fields[j],
                        ExcMessage("Names of compositional fields have to be unique! " + names_of_melt_fields[c] +
                                   " is used more than once."));
            }
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
  namespace InitialComposition
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(Porosity,
                                              "porosity",
                                              "A class that implements initial conditions for the porosity field "
                                              "(or a different field, that could, for example, represent depletion) "
                                              "by computing the equilibrium melt fraction for the given initial "
                                              "condition and reference pressure profile. Note that this plugin only "
                                              "works if there is a compositional field called `porosity' (or whatever "
                                              "names are provided in the parameters section as an alternative), and the "
                                              "used material model implements the 'MeltFractionModel' interface. "
                                              "For all compositional fields except the porosity (or the ones provided "
                                              "in the 'Names of melt fields' subsection) this plugin returns 0.0, "
                                              "and they are therefore not changed as long as the default `add' "
                                              "operator is selected for this plugin.")
  }
}
