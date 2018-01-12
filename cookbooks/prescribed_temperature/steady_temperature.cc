/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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

#include "steady_temperature.h"
//#include <aspect/termination_criteria/steady_temperature.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace TerminationCriteria
  {
    template <int dim>
    bool
    SteadyTemperature<dim>::execute(void)
    {
      const QGauss<dim> quadrature_formula (this->get_fe()
                                            .base_element(this->introspection().base_elements.velocities).degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);
      std::vector<double> temperature_values(n_q_points);

      double local_temperature_integral = 0;

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();
      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
                                                                                         temperature_values);
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                local_temperature_integral += (temperature_values[q] * fe_values.JxW(q));
              }
          }

      const double global_temperature_integral
        = Utilities::MPI::sum (local_temperature_integral, this->get_mpi_communicator());

      // Calculate the average global temperature
      const double average_T = global_temperature_integral / std::sqrt(this->get_volume());

      // Keep a list of times and temperatures at those times
      time_temperature.push_back(std::make_pair(this->get_time(), average_T));

      // If the length of the simulation time covered in the list is shorter than the
      // specified parameter, we must continue the simulation
      const double adjusted_time
        = time_length * (this->convert_output_to_years() ? year_in_seconds : 1);
      if ((time_temperature.size() == 0)
          ||
          (time_temperature.back().first - time_temperature.front().first < adjusted_time))
        return false;

      // Remove old times until we're at the correct time period
      std::list<std::pair<double, double> >::iterator it = time_temperature.begin();
      while (time_temperature.back().first - (*it).first > adjusted_time)
        it++;
      time_temperature.erase(time_temperature.begin(), it);

      // Scan through the list and calculate the min, mean and max temperature
      // We assume a linear change of temperatures between times
      double T_min, T_max, T_prev, time_prev, T_sum=0, T_mean, deviation_max;
      T_min = T_max = T_prev = time_temperature.front().second;
      time_prev = time_temperature.front().first;
      for (it=time_temperature.begin(); it!=time_temperature.end(); ++it)
        {
          T_min = std::min(T_min, (*it).second);
          T_max = std::max(T_max, (*it).second);
          T_sum += (((*it).second + T_prev)/2.0)*((*it).first-time_prev);
          time_prev = (*it).first;
          T_prev = (*it).second;
        }
      T_mean = T_sum/(time_temperature.back().first-time_temperature.front().first);

      // If the min and max are within the acceptable deviation of the mean,
      // we are in steady state and return true, otherwise return false
      deviation_max = std::max(T_mean - T_min, T_max - T_mean);

      // TODO: think about what to do if T_mean = 0
      if (deviation_max/T_mean > relative_deviation)
        return false;

      return true;
    }


    template <int dim>
    void
    SteadyTemperature<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("Steady state temperature");
        {
          prm.declare_entry ("Maximum relative deviation", "0.05",
                             Patterns::Double (0),
                             "The maximum relative deviation of the temperature in recent "
                             "simulation time for the system to be considered in "
                             "steady state. If the actual deviation is smaller "
                             "than this number, then the simulation will be terminated.");
          prm.declare_entry ("Time in steady state", "1e7",
                             Patterns::Double (0),
                             "The minimum length of simulation time that the system "
                             "should be in steady state before termination."
                             "Units: years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "seconds otherwise.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    SteadyTemperature<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Termination criteria");
      {
        prm.enter_subsection("Steady state temperature");
        {
          relative_deviation = prm.get_double ("Maximum relative deviation");
          time_length = prm.get_double ("Time in steady state");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
      AssertThrow (relative_deviation >= 0,
                   ExcMessage("Relative deviation must be greater than or equal to 0."));
      AssertThrow (time_length > 0,
                   ExcMessage("Steady state minimum time period must be greater than 0."));
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace TerminationCriteria
  {
    ASPECT_REGISTER_TERMINATION_CRITERION(SteadyTemperature,
                                          "steady state temperature",
                                          "A criterion that terminates the simulation when the global integral "
                                          "of the temperature field stays within a certain range for a "
                                          "specified period of time.")
  }
}
