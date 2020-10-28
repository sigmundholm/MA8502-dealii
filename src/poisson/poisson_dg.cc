#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
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

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <iostream>

#include "poisson_dg.h"


template<int dim>
PoissonDG<dim>::
PoissonDG(const unsigned int degree,
          const unsigned int n_refines,
          Function<dim> &rhs,
          Function<dim> &bdd_values,
          Function<dim> &analytical_soln)
        : Poisson<dim>(degree, n_refines, rhs, bdd_values, analytical_soln),
          fe(degree) {
    ;
}


template<int dim>
void PoissonDG<dim>::setup_system() {
    this->dof_handler.distribute_dofs(fe);
    std::cout << "  Number of degrees of freedom: "
              << this->dof_handler.n_dofs()
              << std::endl;

    // Find h
    for (const auto &cell : this->dof_handler.active_cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
            double h_k = std::pow(face->measure(), 1.0 / (dim - 1));
            if (h_k > this->h) {
                this->h = h_k;
            }
        }
    }
    std::cout << "  h = " << std::to_string(this->h) << std::endl;

    DoFRenumbering::Cuthill_McKee(this->dof_handler);
    DynamicSparsityPattern dsp(this->dof_handler.n_dofs(),
                               this->dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(this->dof_handler, dsp);

    this->sparsity_pattern.copy_from(dsp);
    this->system_matrix.reinit(this->sparsity_pattern);

    this->solution.reinit(this->dof_handler.n_dofs());
    this->system_rhs.reinit(this->dof_handler.n_dofs());
}


template<int dim>
void PoissonDG<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_v(fe,
                       quadrature_formula,
                       update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_fv(fe,
                            face_quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points |
                            update_normal_vectors | update_JxW_values);

    double mu = 5 / this->h;

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : this->dof_handler.active_cell_iterators()) {
        fe_v.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        // Integrate the contribution from the interior of each cell
        for (const unsigned int q_index : fe_v.quadrature_point_indices()) {
            Point<dim> x_q = fe_v.quadrature_point(q_index);
            for (const unsigned int i : fe_v.dof_indices()) {
                for (const unsigned int j : fe_v.dof_indices()) {
                    cell_matrix(i, j) +=
                            (this->eps *
                             fe_v.shape_grad(i, q_index) *  // grad phi_i(x_q)
                             fe_v.shape_grad(j, q_index) *  // grad phi_j(x_q)
                             fe_v.JxW(q_index));            // dx
                }

                // RHS
                cell_rhs(i) += (fe_v.shape_value(i, q_index) *   // phi_i
                                this->rhs_function->value(x_q) * // f
                                fe_v.JxW(q_index));              // dx
            }
        }

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (const auto &face : cell->face_iterators()) {
            fe_fv.reinit(cell, face);

            for (const unsigned int q_index : fe_fv.quadrature_point_indices()) {
                Point<dim> x_q = fe_fv.quadrature_point(q_index);
                Tensor<1, dim> n = fe_fv.normal_vector(q_index);
                double g_q = this->boundary_values->value(x_q);

                for (const unsigned int i : fe_fv.dof_indices()) {
                    for (const unsigned int j : fe_fv.dof_indices()) {
                        if (face->at_boundary()) {
                            // Boundary face
                            cell_matrix(i, j) +=
                                    (-(n * fe_fv.shape_grad(i, q_index)) *
                                     fe_fv.shape_value(j, q_index)
                                     -
                                     mu *
                                     fe_fv.shape_value(i, q_index) *
                                     (n * fe_fv.shape_grad(j, q_index))
                                    ) * fe_fv.JxW(q_index);
                        } else {
                            // Interior face
                            cell_matrix(i, j) +=
                                    (-n * fe_fv.shape_grad(i, q_index) *
                                     fe_fv.shape_value(j, q_index)
                                    ) * fe_fv.JxW(q_index);
                        }
                    }
                    // Boundary terms for the rhs linear form.
                    if (face->at_boundary()) {
                        cell_rhs(i) += (-g_q * mu *
                                        (n * fe_fv.shape_grad(i, q_index))
                                       ) * fe_fv.JxW(q_index);
                    }
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                this->system_matrix.add(local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i, j));
            }
            this->system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
    /*
    std::map<types::global_dof_index, double> index2bdd_values;
    VectorTools::interpolate_boundary_values(this->dof_handler,
                                             0,
                                             *(this->boundary_values),
                                             index2bdd_values);
    MatrixTools::apply_boundary_values(index2bdd_values,
                                       this->system_matrix,
                                       this->solution,
                                       this->system_rhs);
     */

}


template
class PoissonDG<2>;
