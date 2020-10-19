#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>


#include <iostream>

#include "poisson.h"
#include "rhs.h"

using namespace dealii;


template<int dim>
Poisson<dim>::Poisson(const unsigned int degree,
                      const unsigned int n_refines,
                      Function<dim> &rhs,
                      Function<dim> &bdd_values,
                      Function<dim> &analytical_soln)
        : fe(degree), dof_handler(triangulation), n_refines(n_refines) {
    rhs_function = &rhs;
    boundary_values = &bdd_values;
    analytical_solution = &analytical_soln;
}


template<int dim>
void Poisson<dim>::make_grid() {
    // GridGenerator::cylinder(triangulation, 5, 10);
    Point<dim> p1(-1, -1);
    Point<dim> p2(1, 1);
    GridGenerator::hyper_rectangle(triangulation, p1, p2);
    triangulation.refine_global(n_refines);

    // Write svg of grid to file.
    if (dim == 2) {
        std::ofstream out("poisson-grid.svg");
        GridOut grid_out;
        grid_out.write_svg(triangulation, out);
        std::cout << "Grid written to file as svg." << std::endl;
    }

    std::cout << "  Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

}

template<int dim>
void Poisson<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

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

    DoFRenumbering::Cuthill_McKee(dof_handler);
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template<int dim>
void Poisson<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        // Integrate the contribution from the interior of each cell
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) +=
                            fe_values.shape_grad(i, q_index) *
                            // grad phi_i(x_q)
                            fe_values.shape_grad(j, q_index) *
                            // grad phi_j(x_q)
                            fe_values.JxW(q_index);             // dx
                }

                // RHS
                const auto x_q = fe_values.quadrature_point(q_index);
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                // phi_i(x_q)
                                rhs_function->value(x_q) *         // f(x_q)
                                fe_values.JxW(q_index));             // dx
            }
        }

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    std::map<types::global_dof_index, double> index2bdd_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             *boundary_values,
                                             index2bdd_values);
    MatrixTools::apply_boundary_values(index2bdd_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);

}

template<int dim>
void Poisson<dim>::solve() {
    SparseDirectUMFPACK inverse;
    inverse.initialize(system_matrix);
    inverse.vmult(solution, system_rhs);
}

template<int dim>
void Poisson<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    Vector<double> exact_solution(solution.size());
    VectorTools::interpolate(dof_handler, *analytical_solution, exact_solution);
    Vector<double> diff = exact_solution;
    diff -= solution;
    data_out.add_data_vector(diff, "diff");
    data_out.add_data_vector(exact_solution, "exact");

    data_out.build_patches();
    std::ofstream out("results.vtk");
    data_out.write_vtk(out);
}

template<int dim>
void Poisson<dim>::run() {
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();

    Error error = compute_error();
    std::cout << "    L2 = " << error.l2_error << std::endl;
    std::cout << "    H1 = " << error.h1_error << std::endl;
    std::cout << "    H1-semi = " << error.h1_semi << std::endl;
}


template<int dim>
Error Poisson<dim>::
compute_error() {
    std::cout << "  Compute error" << std::endl;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    double l2_error_integral = 0;
    double h1_semi_error_integral = 0;

    // Numerical solution values and gradients
    std::vector<double> solution_values(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> gradients(fe_values.n_quadrature_points);

    // Exact solution values and gradients
    std::vector<double> exact_solution(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> exact_gradients(fe_values.n_quadrature_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);

        // Numerical solution
        fe_values.get_function_gradients(solution, gradients);
        fe_values.get_function_values(solution, solution_values);

        // Exact solution
        analytical_solution->value_list(fe_values.get_quadrature_points(),
                                        exact_solution);
        analytical_solution->gradient_list(fe_values.get_quadrature_points(),
                                           exact_gradients);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            double diff_values = exact_solution[q] - solution_values[q];
            Tensor<1, dim> diff_grad = exact_gradients[q] - gradients[q];

            l2_error_integral += diff_values * diff_values * fe_values.JxW(q);
            h1_semi_error_integral += diff_grad * diff_grad * fe_values.JxW(q);
        }
    }

    Error error;
    error.mesh_size = h;
    error.l2_error = pow(l2_error_integral, 0.5);
    error.h1_semi = pow(h1_semi_error_integral, 0.5);
    error.h1_error = pow(l2_error_integral + h1_semi_error_integral, 0.5);
    return error;
}


template<int dim>
void Poisson<dim>::
write_header_to_file(std::ofstream &file) {
    file << "mesh_size, e_L2, e_H1, e_H1-semi" << std::endl;
}


template<int dim>
void Poisson<dim>::
write_error_to_file(Error &error, std::ofstream &file) {
    file << error.mesh_size << ","
         << error.l2_error << ","
         << error.l2_error << ","
         << error.h1_semi << std::endl;
}


template
class Poisson<2>;
