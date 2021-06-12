/*
  Copyright (C) 2019 by the authors of the ASPECT code.

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


#ifndef _aspect_termination_criteria_steady_heat_flux_h
#define _aspect_termination_criteria_steady_heat_flux_h

#include <aspect/termination_criteria/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/geometry_model/interface.h>

namespace aspect
{
  namespace TerminationCriteria
  {
    namespace internal
    {
      /**
       * A function that trims the handed over list and removes all entries from the front that are
       * further back in time measured from the last entry than given by the first argument.
       * Additionally it makes sure to always keep two entries in the list, if the list had
       * two or more entries. Otherwise the function does not change the list.
       */
      void trim_time_heat_flux_list (const double necessary_time_in_steady_state,
                                     std::list<std::pair<double, double> > &time_heat_flux_list);
    }

    /**
     * A class that implements a termination criterion based on the steady state
     * of the average heat flux.
     *
     * @ingroup TerminationCriteria
     */
    template <int dim>
    class SteadyHeatFlux : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Evaluate this termination criterion.
         *
         * @return Whether to terminate the simulation (true) or continue
         * (false).
         */
        bool
        execute () override;

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
        double                                  necessary_time_in_steady_state;
        double                                  allowed_relative_deviation;

        /**
         * A set of boundary ids on which the average heat flux will be
         * computed.
         */
        std::set<types::boundary_id> boundary_indicators;

        /**
         * A list of pairs (time, heat flux) that we have computed at
         * previous time steps. This is used to determine when we have reached
         * steady state.
         */
        std::list<std::pair<double, double> >   time_heat_flux;

    };
  }
}

#endif