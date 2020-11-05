#ifndef MA8502_PROJECT_ADVECTION_DIFFUSION_VEC_H
#define MA8502_PROJECT_ADVECTION_DIFFUSION_VEC_H

#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include "rhs_ad_vec.h"


using namespace dealii;


namespace AdvectionDiffusionVector {


    template<int dim>
    class AdvectionDiffusion {
    public:
        AdvectionDiffusion(const unsigned int degree,
                           const unsigned int n_refines,
                           TensorFunction<1, dim> &rhs,
                           TensorFunction<1, dim> &bdd_val,
                           TensorFunction<1, dim> &analytical_soln,
                           unsigned int do_nothing_bdd_id = 1);

        Error run();

        static void write_header_to_file(std::ofstream &file);

        static void write_error_to_file(Error &error, std::ofstream &file);

    protected:
        virtual void make_grid();

        void output_grid();

        void setup_dofs();

        void assemble_system();

        void solve();

        void output_results() const;

        Error compute_error();

        void integrate_cell(const FEValues<dim> &fe_values,
                            double &l2_error_integral_u,
                            double &h1_error_integral_u) const;


        const unsigned int degree;
        const unsigned int n_refines;

        Triangulation<dim> triangulation;
        FESystem<dim> fe;
        DoFHandler<dim> dof_handler;
        TensorFunction<1, dim> *right_hand_side;
        TensorFunction<1, dim> *boundary_values;
        TensorFunction<1, dim> *analytical_solution;
        const unsigned int do_nothing_bdd_id;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;

        double h = 0;
    };

} // namespace AdvectionDiffusionVector


#endif //MA8502_PROJECT_ADVECTION_DIFFUSION_VEC_H
