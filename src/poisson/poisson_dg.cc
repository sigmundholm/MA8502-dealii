#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h>

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
    DoFTools::make_flux_sparsity_pattern(this->dof_handler,
                                         dsp);  // bc tutorial-12
    this->sparsity_pattern.copy_from(dsp);

    this->system_matrix.reinit(this->sparsity_pattern);
    this->solution.reinit(this->dof_handler.n_dofs());
    this->system_rhs.reinit(this->dof_handler.n_dofs());
}


template<int dim>
void PoissonDG<dim>::assemble_system() {

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    // Set the Nitsche penalty parameter. We're using a uniform grid.
    double gamma = 10 * this->degree * (this->degree + 1);
    double mu = gamma / this->h;

    auto cell_worker = [&](const Iterator &cell,
                           ScratchData<dim> &scratch_data,
                           CopyData &copy_data) {
        const unsigned int n_dofs = scratch_data.fe_values.get_fe().dofs_per_cell;
        copy_data.reinit(cell, n_dofs);
        scratch_data.fe_values.reinit(cell);

        const auto &q_points = scratch_data.fe_values.get_quadrature_points();
        const FEValues<dim> &fe_v = scratch_data.fe_values;
        const std::vector<double> &JxW = fe_v.get_JxW_values();

        std::vector<double> f_values(q_points.size());
        this->rhs_function->value_list(q_points, f_values);

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            for (unsigned int i = 0; i < n_dofs; ++i) {
                for (unsigned int j = 0; j < n_dofs; ++j) {
                    copy_data.cell_matrix(i, j) +=
                            (this->eps *
                             fe_v.shape_grad(i, q) *
                             fe_v.shape_grad(j, q)
                            ) * JxW[q];
                }
                copy_data.cell_rhs(i) +=
                        (f_values[q] *
                         fe_v.shape_value(i, q)
                        ) * JxW[q];

            }
        }
    };

    auto boundary_worker = [&](const Iterator &cell,
                               const unsigned int &face_no,
                               ScratchData<dim> &scratch_data,
                               CopyData &copy_data) {
        scratch_data.fe_interface_values.reinit(cell, face_no);
        const FEFaceValuesBase<dim> &fe_face = scratch_data.fe_interface_values.get_fe_face_values(
                0);
        const auto &q_points = fe_face.get_quadrature_points();
        const unsigned int n_face_dofs = fe_face.get_fe().n_dofs_per_cell();
        const std::vector<double> &JxW = fe_face.get_JxW_values();
        const std::vector<Tensor<1, dim>>
                &normals = fe_face.get_normal_vectors();
        std::vector<double> g(q_points.size());
        this->boundary_values->value_list(q_points, g);

        // Set the Nitsche penalty parameter. We're using a uniform grid.
        double gamma = 10 * this->degree * (this->degree + 1);
        double mu = gamma / this->h;

        for (unsigned int q = 0; q < fe_face.n_quadrature_points; ++q) {
            for (unsigned int i = 0; i < n_face_dofs; ++i) {
                for (unsigned int j = 0; j < n_face_dofs; ++j) {
                    copy_data.cell_matrix(i, j) +=
                            this->eps *
                            (-(normals[q] * fe_face.shape_grad(i, q)) *
                             fe_face.shape_value(j, q)
                             -
                             fe_face.shape_value(i, q) *
                             (normals[q] * fe_face.shape_grad(j, q))
                             +
                             mu * fe_face.shape_value(i, q) *
                             fe_face.shape_value(j, q)
                            ) * JxW[q];
                }
                copy_data.cell_rhs(i) +=
                        this->eps *
                        (-g[q] * (normals[q] * fe_face.shape_grad(i, q))
                         +
                         mu * g[q] * fe_face.shape_value(i, q)
                        ) * JxW[q];
            }
        }
    };

    auto face_worker = [&](const Iterator &cell,
                           const unsigned int &f,
                           const unsigned int &sf,
                           const Iterator &ncell,
                           const unsigned int &nf,
                           const unsigned int &nsf,
                           ScratchData<dim> &scratch_data,
                           CopyData &copy_data) {

        FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
        fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

        copy_data.face_data.emplace_back();
        CopyDataFace &copy_data_face = copy_data.face_data.back();
        const unsigned int n_dofs = fe_iv.n_current_interface_dofs();
        copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
        copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

        const std::vector<double> &JxW = fe_iv.get_JxW_values();
        const std::vector<Tensor<1, dim>>
                &normals = fe_iv.get_normal_vectors();

        for (unsigned int q = 0; q < fe_iv.n_quadrature_points; ++q) {
            for (unsigned int i = 0; i < n_dofs; ++i) {
                for (unsigned int j = 0; j < n_dofs; ++j) {
                    copy_data_face.cell_matrix(i, j) +=
                            this->eps *
                            (-(normals[q] * fe_iv.average_gradient(i, q)) *
                             fe_iv.jump(j, q)
                             -
                             fe_iv.jump(i, q) *
                             (normals[q] * fe_iv.average_gradient(j, q))
                             +
                             // This term assures we get continuity over interior faces
                             mu * fe_iv.jump(i, q) *
                             fe_iv.jump(j, q)
                            ) * JxW[q];
                }
            }
        }
    };


    AffineConstraints<double> constraints;
    auto copier = [&](const CopyData &c) {
        constraints.distribute_local_to_global(c.cell_matrix,
                                               c.cell_rhs,
                                               c.local_dof_indices,
                                               this->system_matrix,
                                               this->system_rhs);
        for (auto &cdf : c.face_data) {
            constraints.distribute_local_to_global(cdf.cell_matrix,
                                                   cdf.joint_dof_indices,
                                                   this->system_matrix);
        }
    };

    const unsigned int n_gauss_points = this->dof_handler.get_fe().degree + 1;
    ScratchData<dim> scratch_data(fe, n_gauss_points);
    CopyData copy_data;
    MeshWorker::mesh_loop(this->dof_handler.begin_active(),
                          this->dof_handler.end(),
                          cell_worker, copier, scratch_data, copy_data,
                          MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker, face_worker);

}


template
class PoissonDG<2>;
