/*
  Copyright (C) 2015 - 2019 by the authors of the ASPECT code.

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



#ifndef _aspect_postprocess_post_melt_statistics_h
#define _aspect_postprocess_post_melt_statistics_h

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <deal.II/numerics/data_postprocessor.h>

namespace aspect
{
  namespace Postprocess
  {

    /**
     * A postprocessor that computes some statistics about the
     * melt fraction. This only works for material models that
     * are derived from MaterialModel::MeltFractionModel and
     * implement their own
     * MaterialModel::MeltFractionModel::melt_fraction function.
     *
     * @ingroup Postprocessing
     */
    template <int dim>
    class PostMeltStatistics
      : public Interface<dim>, 
        public ::aspect::SimulatorAccess<dim>,
        public DataPostprocessorScalar<dim>
    {
      public:
        /**
         * Evaluate the melt fraction in the material model.
         */
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;

        void
        evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                              std::vector<Vector<double>> &computed_quantities) const override;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);
        
        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm) override;

      private:
        /**
         * Parameters for anhydrous melting of peridotite after Katz, 2003
         */

        // for the solidus temperature
        double A1;   // °C
        double A2; // °C/Pa
        double A3; // °C/(Pa^2)

        // for the lherzolite liquidus temperature
        double B1;   // °C
        double B2;   // °C/Pa
        double B3; // °C/(Pa^2)

        // for the liquidus temperature
        double C1;   // °C
        double C2;  // °C/Pa
        double C3; // °C/(Pa^2)
      
        // for the reaction coefficient of pyroxene
        double r1;     // cpx/melt
        double r2;     // cpx/melt/GPa
        double M_cpx;  // mass fraction of pyroxenite

        // melt fraction exponent
        double beta;

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
    };
  }
}


#endif
