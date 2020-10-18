#ifndef MA8502_PROJECT_POISSON_H
#define MA8502_PROJECT_POISSON_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "rhs.h"

using namespace dealii;


template<int dim>
class Poisson {
public:
    Poisson(const unsigned int degree,
            const unsigned int n_refines,
            Function<dim> &rhs,
            Function<dim> &bdd_values,
            Function<dim> &analytical_soln);

    void run();

protected:
    void make_grid();

    void setup_system();

    virtual void assemble_system();

    void solve();

    void output_results() const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    unsigned int n_refines;
    double h;

    Function<dim> *rhs_function;
    Function<dim> *boundary_values;
    Function<dim> *analytical_solution;

};


#endif //MA8502_PROJECT_POISSON_H
