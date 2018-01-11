#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/fe/fe_values.h>
#include <aspect/global.h>
#include <aspect/simulator_signals.h>
#include <aspect/geometry_model/interface.h>

namespace aspect
{
  using namespace dealii;

  // Global variables (to be set by parameters)
  bool prescribe_internal_temperatures;

  /**
   * Declare additional parameters.
   */
  void declare_parameters(const unsigned int /*dim*/,
                          ParameterHandler &prm)
  {
    prm.declare_entry ("Prescribe internal temperatures", "false",
                       Patterns::Bool (),
                       "Whether or not to use any prescribed internal velocities. "
                       "Locations in which to prescribe velocities are defined "
                       "in section ``Prescribed velocities/Indicator function'' "
                       "and the velocities are defined in section ``Prescribed "
                       "velocities/Velocity function''. Indicators are evaluated "
                       "at the center of each cell, and all DOFs associated with "
                       "the specified velocity component at the indicated cells "
                       "are constrained."
                      );

  }

  template <int dim>
  void parse_parameters(const Parameters<dim> &,
                        ParameterHandler &prm)
  {
    prescribe_internal_temperatures = prm.get_bool ("Prescribe internal temperatures");
  }

  /**
   * This function is called by a signal which is triggered after the other constraints
   * have been calculated. This enables us to define additional constraints in the mass
   * matrix on any arbitrary degree of freedom in the model space.
   */
  template <int dim>
  void constrain_internal_temperatures (const SimulatorAccess<dim> &simulator_access,
                                      ConstraintMatrix &current_constraints)
  {
    if (prescribe_internal_temperatures)
      {
        const std::vector< Point<dim> > points = simulator_access.get_fe().get_unit_support_points();
        const Quadrature<dim> quadrature (points);
        FEValues<dim> fe_values (simulator_access.get_fe(), quadrature, update_q_points);
        typename DoFHandler<dim>::active_cell_iterator cell;

        // Loop over all cells
        for (cell = simulator_access.get_dof_handler().begin_active();
             cell != simulator_access.get_dof_handler().end();
             ++cell)
          if (! cell->is_artificial())
            {
              fe_values.reinit (cell);
              std::vector<unsigned int> local_dof_indices(simulator_access.get_fe().dofs_per_cell);
              cell->get_dof_indices (local_dof_indices);

              for (unsigned int q=0; q<quadrature.size(); q++)
                // If it's okay to constrain this DOF
                if (current_constraints.can_store_line(local_dof_indices[q]) &&
                    !current_constraints.is_constrained(local_dof_indices[q]))
                  {
                    // Get the temperature component index
                    const unsigned int t_idx =
                      simulator_access.get_fe().system_to_component_index(q).first;

                    // If we're on one of the velocity DOFs
                    if (t_idx ==
                         simulator_access.introspection().component_indices.temperature)
                      {
                        // we get time passed as seconds (always) but may want
                        // to reinterpret it in years
                        const Point<dim> p = fe_values.quadrature_point(q);
                        if (simulator_access.get_geometry_model().depth(p) > 200000)
                          {
                            // Add a constraint of the form dof[q] = T
                            // to the list of constraints.
                            current_constraints.add_line (local_dof_indices[q]);
                            current_constraints.set_inhomogeneity (local_dof_indices[q], 1700);
                          }
                      }
                  }
            }
      }
  }

  // Connect declare_parameters and parse_parameters to appropriate signals.
  void parameter_connector ()
  {
    SimulatorSignals<2>::declare_additional_parameters.connect (&declare_parameters);
    SimulatorSignals<3>::declare_additional_parameters.connect (&declare_parameters);

    SimulatorSignals<2>::parse_additional_parameters.connect (&parse_parameters<2>);
    SimulatorSignals<3>::parse_additional_parameters.connect (&parse_parameters<3>);
  }

  // Connect constraints function to correct signal.
  template <int dim>
  void signal_connector (SimulatorSignals<dim> &signals)
  {
    signals.post_constraints_creation.connect (&constrain_internal_temperatures<dim>);
  }

  // Tell Aspect to send signals to the connector functions
  ASPECT_REGISTER_SIGNALS_PARAMETER_CONNECTOR(parameter_connector)
  ASPECT_REGISTER_SIGNALS_CONNECTOR(signal_connector<2>, signal_connector<3>)
}
