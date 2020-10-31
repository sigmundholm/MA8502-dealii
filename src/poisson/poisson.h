#ifndef MA8502_PROJECT_POISSON_H
#define MA8502_PROJECT_POISSON_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "rhs.h"

using namespace dealii;



struct Error {
    double mesh_size = 0;
    double l2_error = 0;
    double h1_error = 0;
    double h1_semi = 0;
    double sd_error = 0;
};


template<int dim>
class Poisson {
public:
    Poisson(const unsigned int degree,
            const unsigned int n_refines,
            Function<dim> &rhs,
            Function<dim> &bdd_values,
            Function<dim> &analytical_soln);

    Error run();

    Error compute_error();

    static void write_header_to_file(std::ofstream &file);

    static void write_error_to_file(Error &error, std::ofstream &file);

protected:
    virtual void make_grid();

    virtual void setup_system();

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

    unsigned int degree;
    unsigned int n_refines;
    double h = 0;

    double eps = 1;

    Function<dim> *rhs_function;
    Function<dim> *boundary_values;
    Function<dim> *analytical_solution;
    // TensorFunction<1, dim> *vector_field;
};


#endif //MA8502_PROJECT_POISSON_H
