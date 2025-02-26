/*
  Copyright (C) 2015 - 2023 by the authors of the ASPECT code.

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



#include <aspect/postprocess/post_melt_statistics.h>
//#include <aspect/postprocess/visualization/melt_fraction.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/parameter_handler.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    std::pair<std::string,std::string>
    PostMeltStatistics<dim>::execute (TableHandler &statistics)
    {
      // create a quadrature formula based on the temperature element alone.
      const Quadrature<dim> &quadrature_formula = this->introspection().quadratures.temperature;
      const unsigned int n_q_points = quadrature_formula.size();
      std::vector<std::vector<double>> composition_values (this->n_compositional_fields(),
                                                            std::vector<double> (n_q_points));

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_gradients |
                               update_quadrature_points |
                               update_JxW_values);

      MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points, this->n_compositional_fields());

      std::ostringstream output;
      output.precision(4);

      double local_melt_integral = 0.0;
      double local_min_melt = std::numeric_limits<double>::max();
      double local_max_melt = std::numeric_limits<double>::lowest();

      // compute the integral quantities by quadrature
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // fill material model inputs
            fe_values.reinit (cell);
            in.reinit(fe_values, cell, this->introspection(), this->get_solution());

            // we can only postprocess melt fractions if the material model that is used
            // in the simulation has implemented them
            // otherwise, set them to zero
            std::vector<double> melt_fractions(n_q_points, 0.0);
            std::vector<double> temperature_values(n_q_points);
            //if (MaterialModel::MeltFractionModel<dim>::is_melt_fraction_model(this->get_material_model()))
            //  MaterialModel::MeltFractionModel<dim>::as_melt_fraction_model(this->get_material_model())
            //  .melt_fractions(in, melt_fractions);
            template <int dim>
            void
            PostMeltStatistics<dim>::
            evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                                  std::vector<Vector<double>> &computed_quantities) const
            {
              const unsigned int n_quadrature_points = input_data.solution_values.size();
              Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
              Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
              Assert (input_data.solution_values[0].size() == this->introspection().n_components,           ExcInternalError());

              for (unsigned int q=0; q<n_quadrature_points; ++q)
                {
                  const double pressure    = input_data.solution_values[q][this->introspection().component_indices.pressure];
                  const double temperature = input_data.solution_values[q][this->introspection().component_indices.temperature];
                  std::vector<double> composition(this->n_compositional_fields());

                  for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                    composition[c] = input_data.solution_values[q][this->introspection().component_indices.compositional_fields[c]];

                  // anhydrous melting of peridotite after Katz, 2003
                  const double T_solidus  = A1 + 273.15
                                            + A2 * pressure
                                            + A3 * pressure * pressure;
                  const double T_lherz_liquidus = B1 + 273.15
                                                  + B2 * pressure
                                                  + B3 * pressure * pressure;
                  const double T_liquidus = C1 + 273.15
                                            + C2 * pressure
                                            + C3 * pressure * pressure;

                  // melt fraction for peridotite with clinopyroxene
                  double peridotite_melt_fraction;
                  if (temperature < T_solidus || pressure > 1.3e10)
                    peridotite_melt_fraction = 0.0;
                  else if (temperature > T_lherz_liquidus)
                    peridotite_melt_fraction = 1.0;
                  else
                    peridotite_melt_fraction = std::pow((temperature - T_solidus) / (T_lherz_liquidus - T_solidus),beta);

                  // melt fraction after melting of all clinopyroxene
                  const double R_cpx = r1 + r2 * std::max(0.0, pressure);
                  const double F_max = M_cpx / R_cpx;

                  if (peridotite_melt_fraction > F_max && temperature < T_liquidus)
                    {
                      const double T_max = std::pow(F_max,1/beta) * (T_lherz_liquidus - T_solidus) + T_solidus;
                      peridotite_melt_fraction = F_max + (1 - F_max) * std::pow((temperature - T_max) / (T_liquidus - T_max),beta);
                    }

                  // melting of pyroxenite after Sobolev et al., 2011
                  const double T_melting = D1 + 273.15
                                          + D2 * pressure
                                          + D3 * pressure * pressure;

                  const double discriminant = E1*E1/(E2*E2*4) + (temperature-T_melting)/E2;

                  double pyroxenite_melt_fraction;
                  if (temperature < T_melting || pressure > 1.3e10)
                    pyroxenite_melt_fraction = 0.0;
                  else if (discriminant < 0)
                    pyroxenite_melt_fraction = 0.5429;
                  else
                    pyroxenite_melt_fraction = -E1/(2*E2) - std::sqrt(discriminant);

                  double melt_fraction;
                  if (this->introspection().compositional_name_exists("pyroxenite"))
                    {
                      const unsigned int pyroxenite_index = this->introspection().compositional_index_for_name("pyroxenite");
                      melt_fraction = composition[pyroxenite_index] * pyroxenite_melt_fraction +
                                      (1-composition[pyroxenite_index]) * peridotite_melt_fraction;
                    }
                  else
                    melt_fraction = peridotite_melt_fraction;

                  computed_quantities[q](0) = melt_fraction;
                }
            }
            

            for (unsigned int q=0; q<n_q_points; ++q)
              {
                local_melt_integral += melt_fractions[q] * fe_values.JxW(q);
                local_min_melt       = std::min(local_min_melt, melt_fractions[q]);
                local_max_melt       = std::max(local_max_melt, melt_fractions[q]);
              }

          }

      const double global_melt_integral
        = Utilities::MPI::sum (local_melt_integral, this->get_mpi_communicator());
      double global_min_melt = 0;
      double global_max_melt = 0;

      // now do the reductions that are
      // min/max operations. do them in
      // one communication by multiplying
      // one value by -1
      {
        double local_values[2] = { -local_min_melt, local_max_melt };
        double global_values[2];

        Utilities::MPI::max (local_values, this->get_mpi_communicator(), global_values);

        global_min_melt = -global_values[0];
        global_max_melt = global_values[1];
      }


      // finally produce something for the statistics file
      statistics.add_value ("Minimal melt fraction",
                            global_min_melt);
      statistics.add_value ("Total melt volume",
                            global_melt_integral);
      statistics.add_value ("Maximal melt fraction",
                            global_max_melt);

      // also make sure that the other columns filled by this object
      // all show up with sufficient accuracy and in scientific notation
      {
        const char *columns[] = { "Minimal melt fraction",
                                  "Total melt volume",
                                  "Maximal melt fraction"
                                };
        for (auto &column : columns)
          {
            statistics.set_precision (column, 8);
            statistics.set_scientific (column, true);
          }
      }

      output << global_min_melt << ", "
             << global_melt_integral << ", "
             << global_max_melt;

      return std::pair<std::string, std::string> ("Melt fraction min/total/max:",
                                                  output.str());
    }

    template <int dim>
    void
    PostMeltStatistics<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Post melt statistics");
          {
            prm.declare_entry ("A1", "1085.7",
                               Patterns::Double (),
                               "Constant parameter in the quadratic "
                               "function that approximates the solidus "
                               "of peridotite. "
                               "Units: \\si{\\degreeCelsius}.");
            prm.declare_entry ("A2", "1.329e-7",
                               Patterns::Double (),
                               "Prefactor of the linear pressure term "
                               "in the quadratic function that approximates "
                               "the solidus of peridotite. "
                               "\\si{\\degreeCelsius\\per\\pascal}.");
            prm.declare_entry ("A3", "-5.1e-18",
                               Patterns::Double (),
                               "Prefactor of the quadratic pressure term "
                               "in the quadratic function that approximates "
                               "the solidus of peridotite. "
                               "\\si{\\degreeCelsius\\per\\pascal\\squared}.");
            prm.declare_entry ("B1", "1475.0",
                               Patterns::Double (),
                               "Constant parameter in the quadratic "
                               "function that approximates the lherzolite "
                               "liquidus used for calculating the fraction "
                               "of peridotite-derived melt. "
                               "Units: \\si{\\degreeCelsius}.");
            prm.declare_entry ("B2", "8.0e-8",
                               Patterns::Double (),
                               "Prefactor of the linear pressure term "
                               "in the quadratic function that approximates "
                               "the  lherzolite liquidus used for "
                               "calculating the fraction of peridotite-"
                               "derived melt. "
                               "\\si{\\degreeCelsius\\per\\pascal}.");
            prm.declare_entry ("B3", "-3.2e-18",
                               Patterns::Double (),
                               "Prefactor of the quadratic pressure term "
                               "in the quadratic function that approximates "
                               "the  lherzolite liquidus used for "
                               "calculating the fraction of peridotite-"
                               "derived melt. "
                               "\\si{\\degreeCelsius\\per\\pascal\\squared}.");
            prm.declare_entry ("C1", "1780.0",
                               Patterns::Double (),
                               "Constant parameter in the quadratic "
                               "function that approximates the liquidus "
                               "of peridotite. "
                               "Units: \\si{\\degreeCelsius}.");
            prm.declare_entry ("C2", "4.50e-8",
                               Patterns::Double (),
                               "Prefactor of the linear pressure term "
                               "in the quadratic function that approximates "
                               "the liquidus of peridotite. "
                               "\\si{\\degreeCelsius\\per\\pascal}.");
            prm.declare_entry ("C3", "-2.0e-18",
                               Patterns::Double (),
                               "Prefactor of the quadratic pressure term "
                               "in the quadratic function that approximates "
                               "the liquidus of peridotite. "
                               "\\si{\\degreeCelsius\\per\\pascal\\squared}.");
            prm.declare_entry ("r1", "0.5",
                               Patterns::Double (),
                               "Constant in the linear function that "
                               "approximates the clinopyroxene reaction "
                               "coefficient. "
                               "Units: non-dimensional.");
            prm.declare_entry ("r2", "8e-11",
                               Patterns::Double (),
                               "Prefactor of the linear pressure term "
                               "in the linear function that approximates "
                               "the clinopyroxene reaction coefficient. "
                               "Units: \\si{\\per\\pascal}.");
            prm.declare_entry ("beta", "1.5",
                               Patterns::Double (),
                               "Exponent of the melting temperature in "
                               "the melt fraction calculation. "
                               "Units: non-dimensional.");
            prm.declare_entry ("Mass fraction cpx", "0.15",
                               Patterns::Double (),
                               "Mass fraction of clinopyroxene in the "
                               "peridotite to be molten. "
                               "Units: non-dimensional.");
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
          prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    PostMeltStatistics<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Post melt statistics");
          {
            A1              = prm.get_double ("A1");
            A2              = prm.get_double ("A2");
            A3              = prm.get_double ("A3");
            B1              = prm.get_double ("B1");
            B2              = prm.get_double ("B2");
            B3              = prm.get_double ("B3");
            C1              = prm.get_double ("C1");
            C2              = prm.get_double ("C2");
            C3              = prm.get_double ("C3");
            r1              = prm.get_double ("r1");
            r2              = prm.get_double ("r2");
            beta            = prm.get_double ("beta");
            M_cpx           = prm.get_double ("Mass fraction cpx");
            D1              = prm.get_double ("D1");
            D2              = prm.get_double ("D2");
            D3              = prm.get_double ("D3");
            E1              = prm.get_double ("E1");
            E2              = prm.get_double ("E2");
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
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(PostMeltStatistics,
                                  "post melt statistics",
                                  "A postprocessor that computes some statistics about "
                                  "the melt (volume) fraction." )
  }
}
