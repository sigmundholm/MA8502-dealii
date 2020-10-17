//
// Created by Sigmund Eggen Holm on 17/10/2020.
//

#ifndef MA8502_PROJECT_POISSON_H
#define MA8502_PROJECT_POISSON_H


using namespace dealii;


template<int dim>
class Poisson {
public:
    Poisson(const unsigned int degree);

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
};


#endif //MA8502_PROJECT_POISSON_H
