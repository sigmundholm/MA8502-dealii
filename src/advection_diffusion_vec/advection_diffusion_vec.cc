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

#include "rhs_ad_vec.h"
#include "advection_diffusion_vec.h"


namespace AdvectionDiffusionVector {

    using namespace dealii;


    template<int dim>
    AdvectionDiffusion<dim>::
    AdvectionDiffusion(const unsigned int degree,
                       const unsigned int n_refines,
                       TensorFunction<1, dim> &rhs,
                       TensorFunction<1, dim> &bdd_val,
                       TensorFunction<1, dim> &analytical_soln,
                       const unsigned int do_nothing_bdd_id)
            : degree(degree), n_refines(n_refines),
              fe(FESystem<dim>(FE_Q<dim>(degree),
                               dim)), // u (with dim components)
              dof_handler(triangulation), do_nothing_bdd_id(do_nothing_bdd_id) {
        right_hand_side = &rhs;
        boundary_values = &bdd_val;
        analytical_solution = &analytical_soln;
    }


    template<int dim>
    void AdvectionDiffusion<dim>::make_grid() {
        Point<dim> p1(0, 0);
        Point<dim> p2(1, 1);
        GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
        triangulation.refine_global(n_refines);
    }

    template<int dim>
    void AdvectionDiffusion<dim>::output_grid() {
        // Write svg of grid to file.
        if (dim == 2) {
            std::ofstream out("grid.svg");
            GridOut grid_out;
            grid_out.write_svg(triangulation, out);
            std::cout << "  Grid written to file as svg." << std::endl;
        }
        std::ofstream out_vtk("grid.vtk");
        GridOut grid_out;
        grid_out.write_vtk(triangulation, out_vtk);
        std::cout << "  Grid written to file as vtk." << std::endl;

        std::cout << "  Number of active cells: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    template<int dim>
    void AdvectionDiffusion<dim>::setup_dofs() {
        dof_handler.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler);  // TODO hva gjør denne?

        // Find h
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            for (const auto &face : cell->face_iterators()) {
                double h_k = std::pow(face->measure(), 1.0 / (dim - 1));
                if (h_k > h) {
                    h = h_k;
                }
            }
        }
        std::cout << "  h = " << std::to_string(h) << std::endl;

        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];
        std::cout << "  Number of active cells: "
                  << triangulation.n_active_cells()
                  << std::endl
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
    void AdvectionDiffusion<dim>::assemble_system() {
        system_matrix = 0;
        system_rhs = 0;

        QGauss<dim> quadrature_formula(
                fe.degree + 2);  // TODO degree+1 eller +2?
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

        FEValues <dim> fe_v(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
        FEFaceValues <dim> fe_fv(fe,
                                 face_quadrature_formula,
                                 update_values | update_gradients |
                                 update_quadrature_points |
                                 update_normal_vectors |
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
        // const FEValuesExtractors::Scalar pressure(dim);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<2, dim>> grad_phi(dofs_per_cell);
        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi(dofs_per_cell, Tensor<1, dim>());

        VectorField<dim> vector_field;

        double gamma;
        double mu;
        Tensor<1, dim> normal;
        Point<dim> x_q;
        Tensor<1, dim> b_q;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_v.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;

            // Get the values for the RightHandSide for all quadrature points in this cell.
            right_hand_side->value_list(fe_v.get_quadrature_points(),
                                        rhs_values);

            // Integrate the contribution for each cell
            for (const unsigned int q : fe_v.quadrature_point_indices()) {
                x_q = fe_v.quadrature_point(q);
                b_q = vector_field.value(x_q);

                for (const unsigned int k : fe_v.dof_indices()) {
                    grad_phi[k] = fe_v[velocities].gradient(k, q);
                    phi[k] = fe_v[velocities].value(k, q);
                }

                for (const unsigned int i : fe_v.dof_indices()) {
                    for (const unsigned int j : fe_v.dof_indices()) {
                        local_matrix(i, j) +=
                                (scalar_product(grad_phi[j],
                                                grad_phi[i]) // (∇u, ∇v)
                                 +
                                 grad_phi[j] * b_q
                                 * phi[i]
                                ) * fe_v.JxW(q);          // dx
                    }
                    // RHS
                    local_rhs(i) += (rhs_values[q] * phi[i]) * // (f, v)
                                    fe_v.JxW(q);               // dx
                }
            }


            for (const auto &face : cell->face_iterators()) {

                if (face->at_boundary()) {
                    fe_fv.reinit(cell, face);

                    // Evaluate the boundary function for all quadrature points on this face.
                    boundary_values->value_list(fe_fv.get_quadrature_points(),
                                                bdd_values);
                    gamma = 10 * degree * (degree + 1);
                    mu = gamma / h;  // Penalty parameter

                    for (unsigned int q : fe_fv.quadrature_point_indices()) {
                        x_q = fe_fv.quadrature_point(q);
                        b_q = vector_field.value(x_q);
                        normal = fe_fv.normal_vector(q);

                        for (const unsigned int k : fe_fv.dof_indices()) {
                            grad_phi[k] = fe_fv[velocities].gradient(k, q);
                            phi[k] = fe_fv[velocities].value(k, q);
                        }

                        for (const unsigned int i : fe_fv.dof_indices()) {
                            for (const unsigned int j : fe_fv.dof_indices()) {

                                local_matrix(i, j) +=
                                        (-(grad_phi[j] * normal)
                                         * phi[i]                // -(n ∇u, v)
                                         -
                                         phi[j] *
                                         (grad_phi[i] * normal)  // -(u, n ∇v)
                                         +
                                         mu * phi[j] * phi[i]    // μ(u, v)
                                         -
                                         b_q * normal *
                                         phi[j] * phi[i]
                                        ) *
                                        fe_fv.JxW(q);            // ds
                            }

                            local_rhs(i) +=
                                    (-bdd_values[q] *
                                     (grad_phi[i] * normal)      // -(g, n ∇v)
                                     +
                                     mu * bdd_values[q] * phi[i] // μ(g, v)
                                     -
                                     b_q * normal *
                                     bdd_values[q] * phi[i]
                                    ) * fe_fv.JxW(q);            // ds
                        }
                    }
                }
            }

            std::vector<types::global_dof_index> local_dof_indices(
                    dofs_per_cell);
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
    void AdvectionDiffusion<dim>::solve() {
        std::cout << "  Solving the system." << std::endl;
        SparseDirectUMFPACK inverse;
        inverse.initialize(system_matrix);
        inverse.vmult(solution, system_rhs);
    }

    template<int dim>
    void AdvectionDiffusion<dim>::output_results() const {
        // TODO se også Handling VVP.
        // see step-22
        std::vector<std::string> solution_names(dim, "velocity");
        // solution_names.emplace_back("pressure");

        std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(
                dim, DataComponentInterpretation::component_is_part_of_vector);

        // dci.push_back(DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, solution_names,
                                 DataOut<dim>::type_dof_data, dci);

        data_out.build_patches();
        std::ofstream output("results-d" + std::to_string(degree) + "r" +
                             std::to_string(n_refines) + ".vtk");
        data_out.write_vtk(output);
        std::cout << "  Output written to .vtk file." << std::endl;
    }


    template<int dim>
    Error AdvectionDiffusion<dim>::
    compute_error() {
        QGauss<dim> quadrature_formula(
                fe.degree + 2);  // TODO degree+1 eller +2?

        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

        double l2_error_integral = 0;
        double h1_error_integral = 0;

        // Loop through all cells and calculate the norms.
        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            integrate_cell(fe_values, l2_error_integral, h1_error_integral);
        }

        Error error;
        error.mesh_size = h;
        error.l2_error = pow(l2_error_integral, 0.5);
        error.h1_error = pow(l2_error_integral + h1_error_integral, 0.5);
        return error;
    }


    template<int dim>
    void AdvectionDiffusion<dim>::
    integrate_cell(const FEValues<dim> &fe_values,
                   double &l2_error_integral,
                   double &h1_error_integral) const {

        const FEValuesExtractors::Vector velocities(0);

        // Extract the solution values and gradients from the solution vector.
        std::vector<Tensor<1, dim>> solution_values(
                fe_values.n_quadrature_points);
        std::vector<Tensor<2, dim>> gradients(fe_values.n_quadrature_points);

        fe_values[velocities].get_function_values(this->solution,
                                                  solution_values);
        fe_values[velocities].get_function_gradients(this->solution, gradients);

        // Exact solution
        std::vector<Tensor<1, dim>> exact_solution(
                fe_values.n_quadrature_points,
                Tensor<1, dim>());
        analytical_solution->value_list(fe_values.get_quadrature_points(),
                                        exact_solution);

        // TODO calculate the gradient in the analytical solution too, for H1 norm.
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            Tensor<1, dim> diff = exact_solution[q] - solution_values[q];

            l2_error_integral += diff * diff * fe_values.JxW(q);
        }
    }


    template<int dim>
    void AdvectionDiffusion<dim>::
    write_header_to_file(std::ofstream &file) {
        file << "h, e_L2, e_H1" << std::endl;
    }


    template<int dim>
    void AdvectionDiffusion<dim>::
    write_error_to_file(Error &error, std::ofstream &file) {
        file << error.mesh_size << ","
             << error.l2_error << ","
             << error.h1_error << std::endl;
    }


    template<int dim>
    Error AdvectionDiffusion<dim>::run() {
        make_grid();
        output_grid();
        setup_dofs();
        assemble_system();
        solve();
        output_results();
        return compute_error();
    }


    // Initialise the templates.
    template
    class AdvectionDiffusion<2>;

} // namespace AdvectionDiffusionVector
