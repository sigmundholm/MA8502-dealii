#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>

#include "rhs_st.h"

using namespace dealii;


namespace Stokes {


    template<int dim>
    class Stokes {
    public:
        Stokes(const unsigned int degree,
               const int n_refines,
               TensorFunction<1, dim> &rhs,
               TensorFunction<1, dim> &bdd_val,
               TensorFunction<1, dim> &analytical_u,
               Function<dim> &analytical_p,
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

        void integrate_cell(const FEValues<dim, dim> &fe_values,
                            double &u_l2_error_integral,
                            double &u_h1_error_integral,
                            double &p_l2_error_integral,
                            double &p_h1_error_integral,
                            const double &mean_num_pressure,
                            const double &mean_ext_pressure) const;

        const unsigned int degree;
        const int n_refines;
        Triangulation<dim> triangulation;
        FESystem<dim> fe;
        DoFHandler<dim> dof_handler;

        TensorFunction<1, dim> *right_hand_side;
        TensorFunction<1, dim> *boundary_values;
        TensorFunction<1, dim> *analytical_velocity;
        Function<dim> *analytical_pressure;

        const unsigned int do_nothing_bdd_id;
        double h = 0;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
    };
}
