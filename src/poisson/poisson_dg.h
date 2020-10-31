#ifndef MA8502_PROJECT_POISSON_DG_H
#define MA8502_PROJECT_POISSON_DG_H

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include "deal.II/fe/fe_dgq.h"

#include "poisson.h"


template<int dim>
struct ScratchData {
    ScratchData(const FiniteElement<dim> &fe,
                const unsigned int quadrature_degreee,
                const UpdateFlags update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                update_values |
                update_gradients |
                update_quadrature_points |
                update_JxW_values |
                update_normal_vectors)
            : fe_values(fe, QGauss<dim>(quadrature_degreee),
                        update_flags),
              fe_interface_values(fe,
                                  QGauss<dim - 1>(quadrature_degreee),
                                  interface_update_flags) {}

    ScratchData(const ScratchData<dim> &scratch_data)
            : fe_values(scratch_data.fe_values.get_fe(),
                        scratch_data.fe_values.get_quadrature(),
                        scratch_data.fe_values.get_update_flags()),
              fe_interface_values(
                      scratch_data.fe_values
                              .get_mapping(), // TODO: implement for fe_interface_values
                      scratch_data.fe_values.get_fe(),
                      scratch_data.fe_interface_values.get_quadrature(),
                      scratch_data.fe_interface_values.get_update_flags()) {}

    FEValues<dim> fe_values;
    FEInterfaceValues<dim> fe_interface_values;
};


struct CopyDataFace {
    FullMatrix<double> cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
};


struct CopyData {
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace> face_data;

    template<class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell) {
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_rhs.reinit(dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
    }
};


template<int dim>
class PoissonDG : public Poisson<dim> {
public:
    PoissonDG(const unsigned int degree,
              const unsigned int n_refines,
              Function<dim> &rhs,
              Function<dim> &bdd_values,
              Function<dim> &analytical_soln);

protected:
    void setup_system() override;

    void assemble_system() override;

    FE_DGQ<dim> fe;
};


#endif //MA8502_PROJECT_POISSON_DG_H
