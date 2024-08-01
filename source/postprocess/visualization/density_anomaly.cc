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


#include <aspect/postprocess/visualization/density_anomaly.h>
#include <aspect/lateral_averaging.h>
#include <aspect/geometry_model/interface.h>



namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      DensityAnomaly<dim>::
      DensityAnomaly ()
        :
        DataPostprocessorScalar<dim> ("density_anomaly",
                                      update_values | update_quadrature_points ),
        Interface<dim>("kg/m^3")
      {}



      template <int dim>
      void
      DensityAnomaly<dim>::
      update ()
      {
        std::vector<double> density_depth_average(n_slices);
        this->get_lateral_averaging().get_density_averages(density_depth_average);

        // Estimates of the lateral temperature average at each depth are required
        // for all cell depths, including those
        // shallower than the midpoint of the first slice or
        // deeper than the midpoint of the last slice.

        padded_density_depth_average.resize(n_slices+2);

        padded_density_depth_average[0] = 2.*density_depth_average[0] - density_depth_average[1];
        padded_density_depth_average[n_slices+1] = 2.*density_depth_average[n_slices-1] - density_depth_average[n_slices-2];

        std::copy ( density_depth_average.begin(), density_depth_average.end(), padded_density_depth_average.begin() + 1 );

      }



      template <int dim>
      void
      DensityAnomaly<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        const double max_depth = this->get_geometry_model().maximal_depth();
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,           ExcInternalError());
        
        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection());
        MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
                                                     this->n_compositional_fields());

        this->get_material_model().evaluate(in, out);
        if (this->get_parameters().material_averaging != MaterialModel::MaterialAveraging::AveragingOperation::project_to_Q1
            &&
            this->get_parameters().material_averaging != MaterialModel::MaterialAveraging::AveragingOperation::project_to_Q1_only_viscosity)
          MaterialModel::MaterialAveraging::average (this->get_parameters().material_averaging,
                                                     input_data.template get_cell<dim>(),
                                                     Quadrature<dim>(),
                                                     this->get_mapping(),
                                                     in.requested_properties,
                                                     out);

        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            const double density = out.densities[q];
            const double depth = this->get_geometry_model().depth (input_data.evaluation_points[q]);
            // calculate the depth-average cell containing this point. Note that cell centers are offset +0.5 cells in depth.
            const double slice_depth = (depth*n_slices)/max_depth + 0.5;
            const unsigned int idx = static_cast<unsigned int>(slice_depth);
            const double fractional_slice = slice_depth - static_cast<double>(idx);
            Assert(idx<n_slices+1, ExcInternalError());
            const double depth_average_density= (1. - fractional_slice)*padded_density_depth_average[idx] + fractional_slice*padded_density_depth_average[idx+1];
            computed_quantities[q](0) = density - depth_average_density;
          }
      }



      template <int dim>
      void
      DensityAnomaly<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Visualization");
          {
            prm.enter_subsection("Density anomaly");
            {
              prm.declare_entry ("Number of depth slices","20",
                                 Patterns::Integer (1),
                                 "Number of depth slices used to define "
                                 "average density.");

            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }



      template <int dim>
      void
      DensityAnomaly<dim>::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Visualization");
          {
            prm.enter_subsection("density_anomaly");
            {
              n_slices = prm.get_integer("Number of depth slices");
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(DensityAnomaly,
                                                  "density anomaly",
                                                  "A visualization output postprocessor that outputs the temperature minus the depth-average of the temperature."
                                                  "The average temperature is calculated using the lateral averaging function from the ``depth average'' "
                                                  "postprocessor and interpolated linearly between the layers specified through ``Number of depth slices''."
                                                  "\n\n"
                                                  "Physical units: \\si{\\kg/m^3}.")
    }
  }
}
