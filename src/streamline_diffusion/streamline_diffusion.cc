#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <iostream>

#include "../advection_diffusion/rhs_ad.h"
#include "streamline_diffusion.h"


using namespace dealii;


template<int dim>
StreamlineDiffusion<dim>::
StreamlineDiffusion(const unsigned int degree,
                    const unsigned int n_refines,
                    const double eps,
                    const int rho,
                    Function<dim> &rhs,
                    Function<dim> &bdd_values,
                    Function<dim> &analytical_soln)
        : Poisson<dim>(degree, n_refines, rhs, bdd_values, analytical_soln),
          rho(rho) {
    this->eps = eps;

}


template<int dim>
void StreamlineDiffusion<dim>::assemble_system() {
    QGauss<dim> quadrature_formula((this->fe).degree + 1);

    VectorField<dim> vector_field;

    FEValues<dim> fe_values(this->fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_hessians | update_quadrature_points |
                            update_JxW_values);

    const unsigned int dofs_per_cell = (this->fe).dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    double peclet_number = 1 * this->h / this->eps;
    double delta_0 = 0.01;
    double delta_T = 0;

    // Constants for Streamline diffusion (rho = 0)
    if (peclet_number <= 1) {
        delta_T = delta_0 * this->h / 1;
    } else {
        delta_T = delta_0 * pow(this->h, 2) / this->eps;
    }

    // TODO need constants for rho = 1, and -1 too

    for (const auto &cell : this->dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        // Integrate the contribution from the interior of each cell
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            Point<dim> x_q = fe_values.quadrature_point(q_index);
            Tensor<1, dim> b_q = vector_field.value(x_q);
            double f_q = this->rhs_function->value(x_q);

            for (const unsigned int i : fe_values.dof_indices()) {

                Tensor<2, dim> hessian_i = fe_values.shape_hessian(i, q_index);
                double laplacian_i = trace(hessian_i);

                for (const unsigned int j : fe_values.dof_indices()) {
                    Tensor<2, dim> hessian_j = fe_values.shape_hessian(j,
                                                                       q_index);
                    double laplacian_j = trace(hessian_j);

                    cell_matrix(i, j) +=
                            (this->eps * fe_values.shape_grad(j, q_index)
                             * fe_values.shape_grad(i, q_index)
                             +
                             (b_q * fe_values.shape_grad(j, q_index))
                             * fe_values.shape_value(i, q_index)
                             +
                             delta_T
                             * ((-this->eps * laplacian_j
                                 + (b_q * fe_values.shape_grad(j, q_index))
                                ) * (-this->eps * laplacian_i
                                     + rho *
                                       (b_q * fe_values.shape_grad(i, q_index)))
                             )
                            ) * fe_values.JxW(q_index);            // dx
                }

                // RHS
                cell_rhs(i) += (fe_values.shape_value(i, q_index)  // phi_i
                                * f_q                              // f(x_q)
                                +
                                delta_T * f_q
                                * (-this->eps * laplacian_i
                                   + rho *
                                     (b_q * fe_values.shape_grad(i, q_index)))
                               ) * fe_values.JxW(q_index);             // dx
            }
        }

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                this->system_matrix.add(local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i, j));
            }
            this->system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    std::map<types::global_dof_index, double> index2bdd_values;
    VectorTools::interpolate_boundary_values(this->dof_handler,
                                             0,
                                             *(this->boundary_values),
                                             index2bdd_values);
    MatrixTools::apply_boundary_values(index2bdd_values,
                                       this->system_matrix,
                                       this->solution,
                                       this->system_rhs);

}

template
class StreamlineDiffusion<2>;
