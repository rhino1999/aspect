/*
  Copyright (C) 2011 - 2024 by the authors of the ASPECT code.

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


#include <aspect/postprocess/melt_production_statistics.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/postprocess/visualization/melt_fraction.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    MeltProductionStatistics<dim>::execute (TableHandler &statistics)
    {
      if (depletion_field_names.size() == 0)
        return {"", ""};

      // create a quadrature formula based on the compositional element alone.
      const Quadrature<dim> &quadrature_formula = this->introspection().quadratures.compositional_field_max;
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);

      // Store the temperature and pressure at each quadrature point for the current time step.
      std::vector<double> temperatures (n_q_points);
      std::vector<double> pressures (n_q_points);
      // Store the depletion of each melting composition at each quadrature point for the current time step.
      std::vector<std::vector<double>> second_melting_composition_depletion((depletion_field_names.size()-1),n_q_points);
      std::vector<double> peridotite_depletion(n_q_points);

      std::vector<double> melting_composition_depletion_index((depletion_field_names.size()-1));
      for (unsigned int mc=0; mc<(depletion_field_names.size()-1); ++mc)
          melting_composition_depletion_index[mc] = this->introspection().compositional_index_for_name(depletion_field_names[mc]);
      
      // Store the depletion of each melting composition at each quadrature point for the previous time step.
      std::vector<std::vector<double>> old_second_melting_composition_depletion ((depletion_field_names.size()-1),n_q_points);
      std::vector<double> old_peridotite_depletion (n_q_points);

      // The first element of the vector is always for peridotite depletion, and the rest are for the other melting compositions.
      std::vector<double> local_depletion_integrals (depletion_field_names.size());

      // Compute the melt mass fraction for current time step using P, T solution values and the melt fraction model.
      // Compare the melt mass fraction at each quadrature point with the corresponding depletion at the previous time 
      // step to compute the melt fraction increase at the current time step.
      // Then compute the integral quantities by quadrature
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);

            const unsigned int peridotite_depletion_index = this->introspection().compositional_index_for_name("peridotite_depletion");
            fe_values[simulator_access.introspection().extractors.temperature].get_function_values (simulator_access.get_solution(), 
                                                                                                    temperatures);
            fe_values[simulator_access.introspection().extractors.pressure].get_function_values (simulator_access.get_solution(), 
                                                                                                 pressures);

            for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
              {
                for (unsigned int mc=1; mc<(depletion_field_names.size()-1); ++mc)
                  {
                    old_second_melting_composition_depletion[mc].resize(n_q_points);
                    fe_values[melting_composition_depletion_index[mc]].get_function_values (simulator_access.get_old_solution(), 
                                                                                            old_second_melting_composition_depletion[mc]);
                  }
                fe_values[peridotite_depletion_index].get_function_values (simulator_access.get_old_solution(), old_peridotite_depletion);
                fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                    compositional_values);
                for (unsigned int q=0; q<n_q_points; ++q)
                  // Loop over all depletion fields to compute the melt fraction increase at each quadrature point.
                  for (unsigned int mc=1; mc<(depletion_field_names.size()-1); ++mc)
                    {
                      // Find the position of "_depletion" and extract the melting model name from the depletion field name
                      size_t pos = depletion_field_names[mc].find("_depletion");
                      if (pos != std::string::npos) 
                          melting_model_names[mc] = depletion_field_names[mc].substr(0, pos); // Extracts everything before "_depletion"
                      else
                          Assert(false, ExcMessage("Could not find '_depletion' in depletion field name."));
                      const double comp_melt_fraction = melt_fraction(temperatures[q], pressures[q], melting_model_names[mc]);
                      second_melting_composition_depletion_change[mc][q] = comp_melt_fraction * compositional_values[q]- old_second_melting_composition_depletion[mc][q];

                    }
                  local_compositional_integrals[c] += compositional_values[q]*fe_values.JxW(q);
              }
          }
      // compute the sum over all processors
      std::vector<double> global_compositional_integrals (local_compositional_integrals.size());
      Utilities::MPI::sum (local_compositional_integrals,
                           this->get_mpi_communicator(),
                           global_compositional_integrals);

      // compute min/max by simply
      // looping over the elements of the
      // solution vector.
      std::vector<double> local_min_compositions (this->n_compositional_fields(),
                                                  std::numeric_limits<double>::max());
      std::vector<double> local_max_compositions (this->n_compositional_fields(),
                                                  std::numeric_limits<double>::lowest());

      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          IndexSet range = this->get_solution().block(this->introspection().block_indices.compositional_fields[c]).locally_owned_elements();
          for (unsigned int i=0; i<range.n_elements(); ++i)
            {
              const unsigned int idx = range.nth_index_in_set(i);
              const double val =  this->get_solution().block(this->introspection().block_indices.compositional_fields[c])(idx);

              local_min_compositions[c] = std::min<double> (local_min_compositions[c], val);
              local_max_compositions[c] = std::max<double> (local_max_compositions[c], val);
            }

        }

      // now do the reductions over all processors
      std::vector<double> global_min_compositions (this->n_compositional_fields(),
                                                   std::numeric_limits<double>::max());
      std::vector<double> global_max_compositions (this->n_compositional_fields(),
                                                   std::numeric_limits<double>::lowest());
      {
        Utilities::MPI::min (local_min_compositions,
                             this->get_mpi_communicator(),
                             global_min_compositions);
        Utilities::MPI::max (local_max_compositions,
                             this->get_mpi_communicator(),
                             global_max_compositions);
      }

      // finally produce something for the statistics file
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          statistics.add_value ("Minimal value for composition " + this->introspection().name_for_compositional_index(c),
                                global_min_compositions[c]);
          statistics.add_value ("Maximal value for composition " + this->introspection().name_for_compositional_index(c),
                                global_max_compositions[c]);
          statistics.add_value ("Global mass for composition " + this->introspection().name_for_compositional_index(c),
                                global_compositional_integrals[c]);
        }

      // also make sure that the other columns filled by this object
      // all show up with sufficient accuracy and in scientific notation
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          const std::string columns[] = { "Minimal value for composition " + this->introspection().name_for_compositional_index(c),
                                          "Maximal value for composition " + this->introspection().name_for_compositional_index(c),
                                          "Global mass for composition " + this->introspection().name_for_compositional_index(c)
                                        };
          for (const auto &col : columns)
            {
              statistics.set_precision (col, 8);
              statistics.set_scientific (col, true);
            }
        }

      std::ostringstream output;
      output.precision(4);
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
        {
          output << global_min_compositions[c] << '/'
                 << global_max_compositions[c] << '/'
                 << global_compositional_integrals[c];
          if (c+1 != this->n_compositional_fields())
            output << " // ";
        }

      return std::pair<std::string, std::string> ("Compositions min/max/mass:",
                                                  output.str());
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(MeltProductionStatistics,
                                  "melt production statistics",
                                  "A postprocessor that computes some statistics about "
                                  "the melt production that is computed using the pressure, "
                                  "temperature, and other relevant fields. This postprocessor "
                                  "only keeps track of the depletion associated with melting "
                                  "and assumes that melting does not affect model's dynamics. ")
  }
}
