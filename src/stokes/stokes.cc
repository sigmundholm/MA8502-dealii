#include <deal.II/base/logstream.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "stokes.h"


namespace Stokes {


    using namespace dealii;


    template<int dim>
    double RightHandSide<dim>::point_value(const Point<dim> &p, const unsigned int) const {
        (void) p;
        return 0;
    }

    template<int dim>
    void RightHandSide<dim>::vector_value(const Point<dim> &p, Tensor<1, dim> &value) const {
        for (unsigned int c = 0; c < dim; ++c)
            value[c] = point_value(p, c);
    }

    template<int dim>
    void RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                                        std::vector<Tensor<1, dim>> &values) const {
        AssertDimension(points.size(), values.size());
        for (unsigned int i = 0; i < values.size(); ++i) {
            vector_value(points[i], values[i]);
        }
    }


    template<int dim>
    double BoundaryValues<dim>::point_value(const Point<dim> &p, const unsigned int component) const {
        (void) p;
        if (component == 0 && p[0] == 0) {
            if (dim == 2) {
                return -2.5 * (p[1] - 0.41) * p[1];
            }
            throw std::exception(); // TODO fix 3D
        }
        return 0;
    }

    template<int dim>
    void BoundaryValues<dim>::vector_value(const Point<dim> &p, Tensor<1, dim> &value) const {
        for (unsigned int c = 0; c < dim; ++c)
            value[c] = point_value(p, c);
    }

    template<int dim>
    void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                         std::vector<Tensor<1, dim>> &values) const {
        AssertDimension(points.size(), values.size());
        for (unsigned int i = 0; i < values.size(); ++i) {
            vector_value(points[i], values[i]);
        }
    }

    template<int dim>
    StokesNitsche<dim>::StokesNitsche(const unsigned int degree, RightHandSide<dim> &rhs, BoundaryValues<dim> &bdd_val,
                                      const unsigned int do_nothing_bdd_id)
            : degree(degree),
              fe(FESystem<dim>(FE_Q<dim>(degree + 1), dim), 1, FE_Q<dim>(degree),
                 1), // u (with dim components), p (scalar component)
              dof_handler(triangulation), do_nothing_bdd_id(do_nothing_bdd_id) {
        right_hand_side = &rhs;
        boundary_values = &bdd_val;
    }


    template<int dim>
    void StokesNitsche<dim>::make_grid() {
        GridGenerator::channel_with_cylinder(triangulation, 0.03, 2, 2.0, true);
        triangulation.refine_global(dim == 2 ? 3 : 0);
    }

    template<int dim>
    void StokesNitsche<dim>::output_grid() {
        // Write svg of grid to file.
        if (dim == 2) {
            std::ofstream out("nitsche-stokes-grid.svg");
            GridOut grid_out;
            grid_out.write_svg(triangulation, out);
            std::cout << "  Grid written to file as svg." << std::endl;
        }
        std::ofstream out_vtk("nitsche-stokes-grid.vtk");
        GridOut grid_out;
        grid_out.write_vtk(triangulation, out_vtk);
        std::cout << "  Grid written to file as vtk." << std::endl;

        std::cout << "  Number of active cells: " << triangulation.n_active_cells() << std::endl;
    }

    template<int dim>
    void StokesNitsche<dim>::setup_dofs() {
        dof_handler.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler);  // TODO hva gjør denne?

        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];
        std::cout << "  Number of active cells: " << triangulation.n_active_cells() << std::endl
                  << "  Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << n_u << " + " << n_p << ')' << std::endl;

        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
    }

    template<int dim>
    void StokesNitsche<dim>::assemble_system() {
        system_matrix = 0;
        system_rhs = 0;

        QGauss<dim> quadrature_formula(fe.degree + 2);  // TODO degree+1 eller +2?
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points | update_normal_vectors |
                                         update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_q_face_points = face_quadrature_formula.size();

        // Matrix and vector for the contribution of each cell
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>>
                rhs_values(n_q_points, Tensor<1, dim>());
        std::vector<Tensor<1, dim>>
                bdd_values(n_q_face_points, Tensor<1, dim>());

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
        std::vector<double> phi_p(dofs_per_cell);

        double h;
        double mu;
        Tensor<1, dim> normal;
        Tensor<1, dim> x_q;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;

            // Get the values for the RightHandSide for all quadrature points in this cell.
            right_hand_side->value_list(fe_values.get_quadrature_points(), rhs_values);

            // Integrate the contribution for each cell
            for (const unsigned int q : fe_values.quadrature_point_indices()) {

                for (const unsigned int k : fe_values.dof_indices()) {
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    phi_u[k] = fe_values[velocities].value(k, q);
                    phi_p[k] = fe_values[pressure].value(k, q);
                }

                for (const unsigned int i : fe_values.dof_indices()) {
                    for (const unsigned int j : fe_values.dof_indices()) {
                        local_matrix(i, j) +=
                                (scalar_product(grad_phi_u[i],
                                                grad_phi_u[j])  // (grad u, grad v)
                                 - (div_phi_u[j] * phi_p[i])    // -(div v, p)
                                 - (div_phi_u[i] * phi_p[j])    // -(div u, q)
                                ) * fe_values.JxW(q);           // dx
                    }
                    // RHS
                    local_rhs(i) +=
                            rhs_values[q] * phi_u[i]  // (f, v)
                            * fe_values.JxW(q);       // dx
                }
            }


            for (const auto &face : cell->face_iterators()) {

                // The right boundary has boundary_id=1, so do nothing there for outflow.
                if (face->at_boundary() && face->boundary_id() != do_nothing_bdd_id) {
                    fe_face_values.reinit(cell, face);

                    // Evaluate the boundary function for all quadrature points on this face.
                    boundary_values->value_list(fe_face_values.get_quadrature_points(), bdd_values);

                    h = std::pow(face->measure(), 1.0 / (dim - 1));
                    mu = 50 / h;  // Penalty parameter

                    for (unsigned int q : fe_face_values.quadrature_point_indices()) {
                        x_q = fe_face_values.quadrature_point(q);
                        normal = fe_face_values.normal_vector(q);

                        for (const unsigned int k : fe_face_values.dof_indices()) {
                            grad_phi_u[k] = fe_face_values[velocities].gradient(k, q);
                            div_phi_u[k] = fe_face_values[velocities].divergence(k, q);
                            phi_u[k] = fe_face_values[velocities].value(k, q);
                            phi_p[k] = fe_face_values[pressure].value(k, q);
                        }

                        for (const unsigned int i : fe_face_values.dof_indices()) {
                            for (const unsigned int j : fe_face_values.dof_indices()) {

                                local_matrix(i, j) +=
                                        (-(grad_phi_u[i] * normal) * phi_u[j]  // -(n * grad u, v)
                                         - (grad_phi_u[j] * normal) * phi_u[i] // -(n * grad v, u)
                                         + mu * (phi_u[i] * phi_u[j])          // mu (u, v)
                                         + (normal * phi_u[j]) * phi_p[i]      // (n * v, p)
                                         + (normal * phi_u[i]) * phi_p[j]      // (n * u, q)
                                        ) * fe_face_values.JxW(q);             // ds
                            }

                            Tensor<1, dim> prod_r = mu * phi_u[i] - grad_phi_u[i] * normal + phi_p[i] * normal;
                            local_rhs(i) +=
                                    prod_r * bdd_values[q]    // (g, mu v - n grad v + q * n)
                                    * fe_face_values.JxW(q);  // ds
                        }
                    }
                }
            }

            std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    system_matrix.add(local_dof_indices[i],
                                      local_dof_indices[j],
                                      local_matrix(i, j));
                }
                system_rhs(local_dof_indices[i]) += local_rhs(i);
            }
        }
    }

    template<int dim>
    void StokesNitsche<dim>::solve() {
        // TODO annen løser? Løs på blokk-form?
        SparseDirectUMFPACK inverse;
        inverse.initialize(system_matrix);
        inverse.vmult(solution, system_rhs);
    }

    template<int dim>
    void StokesNitsche<dim>::output_results() const {
        // TODO se også Handling VVP.
        // see step-22
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        dci.push_back(DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, solution_names, DataOut<dim>::type_dof_data, dci);

        data_out.build_patches();
        std::ofstream output("nitsche-stokes.vtk");
        data_out.write_vtk(output);
        std::cout << "  Output written to .vtk file." << std::endl;
    }

    template<int dim>
    void StokesNitsche<dim>::run() {
        make_grid();
        output_grid();
        setup_dofs();
        assemble_system();
        solve();
        output_results();
    }


    // Initialise the templates.
    template
    class StokesNitsche<2>;

    template
    class RightHandSide<2>;


    template
    class BoundaryValues<2>;

} // namespace Stokes
