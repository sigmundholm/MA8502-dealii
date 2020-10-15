#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
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

#include <fstream>
#include <iostream>

using namespace dealii;

template<int dim>
class PoissonNitsche {
public:
    PoissonNitsche(const unsigned int degree);

    void run();

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void solve();

    void output_results() const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    double eps = 0.01;
};

// Functions for right hand side and boundary values.

template<int dim>
class RightHandSide : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template<int dim>
class BoundaryValues : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
};


template<int dim>
double
RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 1;
}

template<int dim>
double
BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 0;
}


template<int dim>
class VectorField : public TensorFunction<1, dim> {
public:
    Tensor<1, dim> value(const Point<dim> &p) const override;
};


template<int dim>
Tensor<1, dim> VectorField<dim>::
value(const Point<dim> &p) const {
    (void) p;
    Tensor<1, dim> value;
    value[0] = 0;
    value[1] = 1;
    return value;
}


template<int dim>
PoissonNitsche<dim>::PoissonNitsche(const unsigned int degree)
        : fe(degree), dof_handler(triangulation) {}


template<int dim>
void PoissonNitsche<dim>::make_grid() {
    // GridGenerator::cylinder(triangulation, 5, 10);
    Point<dim> p1(-1, -1);
    Point<dim> p2(1, 1);
    GridGenerator::hyper_rectangle(triangulation, p1, p2);
    triangulation.refine_global(7);

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
void PoissonNitsche<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    DoFRenumbering::Cuthill_McKee(dof_handler);
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template<int dim>
void PoissonNitsche<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    RightHandSide<dim> right_hand_side;
    VectorField<dim> vector_field;

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
            Point<dim> x_q = fe_values.quadrature_point(q_index);
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) +=
                            (eps * fe_values.shape_grad(i, q_index)
                             * fe_values.shape_grad(j, q_index)
                             +
                             (vector_field.value(x_q)
                              * fe_values.shape_grad(i, q_index))
                             * fe_values.shape_value(j, q_index)
                            ) * fe_values.JxW(q_index);            // dx
                }

                // RHS
                const auto x_q = fe_values.quadrature_point(q_index);
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *  // phi_i
                                right_hand_side.value(x_q) *         // f(x_q)
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

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);

}

template<int dim>
void PoissonNitsche<dim>::solve() {
    SparseDirectUMFPACK inverse;
    inverse.initialize(system_matrix);
    inverse.vmult(solution, system_rhs);
}

template<int dim>
void PoissonNitsche<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream out(dim == 2 ? "poisson-2d.vtk" : "poisson-3d.vtk");
    data_out.write_vtk(out);
}

template<int dim>
void PoissonNitsche<dim>::run() {
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
}

int main() {
    std::cout << "PoissonNitsche" << std::endl;
    {
        PoissonNitsche<2> poissonNitsche(1);
        poissonNitsche.run();
    }
}