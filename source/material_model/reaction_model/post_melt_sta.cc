/*
  Copyright (C) 2024 by the authors of the ASPECT code.

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


#include <aspect/material_model/reaction_model/post_melt_sta.h>
#include <aspect/utilities.h>
#include <aspect/material_model/reaction_model/katz2003_mantle_melting.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <deal.II/base/parameter_handler.h>


namespace aspect
{
  namespace MaterialModel
  {
    namespace ReactionModel
    {
      template <int dim>
      void
      PostMeltSta<dim>::calculate_reaction_terms (const typename Interface<dim>::MaterialModelInputs  &in,
                                                typename Interface<dim>::MaterialModelOutputs       &out) const
      {
        for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
          {
            const double depth = this->get_geometry_model().depth(in.position[i]);
            const double pressure    = this->get_adiabatic_conditions().pressure(in.position[i]);//eos_in.pressure[i];
            const double temperature = in.temperature[i];
            // compute melt fraction
            const double old_pyrolite_depletion = in.composition[i][pyrolite_depletion_index];
            const double old_pyroxenite_depletion = in.composition[i][pyroxenite_depletion_index];
            double pyrolite_melt_fraction = katz2003_model.melt_fraction(temperature, pressure) * (1-old_pyroxenite_depletion-in.composition[i][pyroxenite_index]);
            
            // melting of pyroxenite after Sobolev et al., 2011
            const double T_melting = D1 + 273.15
                                    + D2 * pressure
                                    + D3 * pressure * pressure;

            const double discriminant = E1*E1/(E2*E2*4) + (temperature-T_melting)/E2;

            double pyroxenite_melt_fraction;
            if (temperature < T_melting || pressure > 1.3e10)
              pyroxenite_melt_fraction = 0.0;
            else if (discriminant < 0)
              pyroxenite_melt_fraction = 0.5429 * (old_pyroxenite_depletion+in.composition[i][pyroxenite_index]);
            else
              pyroxenite_melt_fraction = (-E1/(2*E2) - std::sqrt(discriminant)) *  (old_pyroxenite_depletion+in.composition[i][pyroxenite_index]);
            
            double pyrolite_melt_fraction_change;
            if (pyrolite_melt_fraction > old_pyrolite_depletion)
              {
                pyrolite_melt_fraction_change = pyrolite_melt_fraction - old_pyrolite_depletion;
              }
            else
              pyrolite_melt_fraction_change = 0.0;

            pyrolite_melt_fraction_change = std::max(pyrolite_melt_fraction_change, 0.0);

            double pyroxenite_melt_fraction_change;
            if (pyroxenite_melt_fraction > old_pyroxenite_depletion)
              {
                pyroxenite_melt_fraction_change = pyroxenite_melt_fraction - old_pyroxenite_depletion;
              }
            else
              pyroxenite_melt_fraction_change = 0.0;
            pyroxenite_melt_fraction_change = std::max(pyroxenite_melt_fraction_change, 0.0);
            
            // Do not change other reaction terms.
            // There might be other reactions computed in the material model
            // (including for the basalt and harzburgite fields).
            out.reaction_terms[i][pyroxenite_index] = -pyroxenite_melt_fraction_change;
            out.reaction_terms[i][pyroxenite_depletion_index] = pyroxenite_melt_fraction_change;
            out.reaction_terms[i][pyrolite_depletion_index] = pyrolite_melt_fraction_change;      
          }
      }



      template <int dim>
      void
      PostMeltSta<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.declare_entry ("D1", "976.0",
                            Patterns::Double (),
                            "Constant parameter in the quadratic "
                            "function that approximates the solidus "
                            "of pyroxenite. "
                            "Units: \\si{\\degreeCelsius}.");
        prm.declare_entry ("D2", "1.329e-7",
                            Patterns::Double (),
                            "Prefactor of the linear pressure term "
                            "in the quadratic function that approximates "
                            "the solidus of pyroxenite. "
                            "Note that this factor is different from the "
                            "value given in Sobolev, 2011, because they use "
                            "the potential temperature whereas we use the "
                            "absolute temperature. "
                            "\\si{\\degreeCelsius\\per\\pascal}.");
        prm.declare_entry ("D3", "-5.1e-18",
                            Patterns::Double (),
                            "Prefactor of the quadratic pressure term "
                            "in the quadratic function that approximates "
                            "the solidus of pyroxenite. "
                            "\\si{\\degreeCelsius\\per\\pascal\\squared}.");
        prm.declare_entry ("E1", "663.8",
                            Patterns::Double (),
                            "Prefactor of the linear depletion term "
                            "in the quadratic function that approximates "
                            "the melt fraction of pyroxenite. "
                            "\\si{\\degreeCelsius\\per\\pascal}.");
        prm.declare_entry ("E2", "-611.4",
                            Patterns::Double (),
                            "Prefactor of the quadratic depletion term "
                            "in the quadratic function that approximates "
                            "the melt fraction of pyroxenite. "
                            "\\si{\\degreeCelsius\\per\\pascal\\squared}.");
      }



      template <int dim>
      void
      PostMeltSta<dim>::parse_parameters (ParameterHandler &prm)
      {
        D1              = prm.get_double ("D1");
        D2              = prm.get_double ("D2");
        D3              = prm.get_double ("D3");
        E1              = prm.get_double ("E1");
        E2              = prm.get_double ("E2");

        AssertThrow(this->introspection().compositional_name_exists("pyroxenite") &&
                    this->introspection().compositional_name_exists("pyroxenite_depletion") &&
                    this->introspection().compositional_name_exists("pyrolite_depletion"),
                    ExcMessage("The reaction model <post melt statistics> "
                               "can only be used if there is a compositional field named "
                               "'pyroxenite', 'pyroxenite_depletion', and 'pyrolite_depletion'."));

        pyroxenite_index = this->introspection().compositional_index_for_name("pyroxenite");
        pyrolite_depletion_index = this->introspection().compositional_index_for_name("pyrolite_depletion");
        pyroxenite_depletion_index = this->introspection().compositional_index_for_name("pyroxenite_depletion");
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    namespace ReactionModel
    {
#define INSTANTIATE(dim) \
  template class PostMeltSta<dim>;

      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
