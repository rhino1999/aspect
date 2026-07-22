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

#ifndef _aspect_material_model_reaction_model_post_melt_sta_h
#define _aspect_material_model_reaction_model_post_melt_sta_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/reaction_model/katz2003_mantle_melting.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    namespace ReactionModel
    {

      /**
      * A simplified model to calculate the change in composition upon melting
      * of average mantle as it approaches the surface to produce a basaltic crust
      * and a harzburgitic lithosphere. The model assumes that the crust is
      * generated at a constant depth, and that the lithosphere is generated
      * below the crust at a constant depth. The reaction producing crust and
      * lithosphere only occurs in material that is upwelling, but does not take
      * into account the temperature of the upwelling material.
      *
      * @ingroup ReactionModel
      */
      template <int dim>
      class PostMeltSta : public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
          * Declare the parameters this function takes through input files.
          */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm);

          /**
           * Compute the change in composition for the pyroxenite and pyrolite chemical
           * fields upon melting. The reaction terms are
           * computed for as many points as are provided in @p in and they are stored
           * in the material model outputs object @p out.
           */
          void
          calculate_reaction_terms (const typename Interface<dim>::MaterialModelInputs  &in,
                                    typename Interface<dim>::MaterialModelOutputs       &out) const;

        private:

          /**
           * Parameters for melting of pyroxenite after Sobolev et al., 2011
           */

          // for the melting temperature
          double D1;    // °C
          double D2;  // °C/Pa
          double D3; // °C/(Pa^2)

          // for the melt-fraction dependence of productivity
          double E1;
          double E2;

          /*
          * Object for computing the melt parameters
          */
          ReactionModel::Katz2003MantleMelting<dim> katz2003_model;

          /**
           * The indices of the compositional fields that store the pyroxenite and
           * pyrolite chemical compositions.
           */
          unsigned int pyroxenite_index;
          unsigned int pyrolite_depletion_index;
          unsigned int pyroxenite_depletion_index;
      };
    }

  }
}

#endif
