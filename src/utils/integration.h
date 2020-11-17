#ifndef MA8502_PROJECT_INTEGRATION_H
#define MA8502_PROJECT_INTEGRATION_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>

using namespace dealii;


namespace Utils {

    template<int dim, int spacedim>
    void compute_mean_pressure(DoFHandler<dim, spacedim> &dof,
                               FEValues<dim> &fe_v,
                               Vector<double> &solution,
                               Function<dim> &pressure,
                               double &mean_numerical_pressure,
                               double &mean_analytical_pressure) {
        Assert(solution.size() == dof.n_dofs(),
               ExcDimensionMismatch(solution.size(), dof.n_dofs()));

        // TODO Take Quadrature as argument instead of FEValues, and
        //  create FEValues here, as is done in VectorTools:compute_mean_value

        const FEValuesExtractors::Scalar p(dim);
        std::vector<double> numerical(fe_v.n_quadrature_points);
        std::vector<double> exact(fe_v.n_quadrature_points);

        // Loop through all cells and calculate the norms.
        double area = 0;
        for (const auto &cell : dof.active_cell_iterators()) {
            fe_v.reinit(cell);

            // Extract the numerical pressure values from the solution vector.
            fe_v[p].get_function_values(solution, numerical);

            // Extract the exact pressure values
            pressure.value_list(fe_v.get_quadrature_points(), exact);

            for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
                mean_numerical_pressure += numerical[q] * fe_v.JxW(q);
                mean_analytical_pressure += exact[q] * fe_v.JxW(q);
                area += fe_v.JxW(q);
            }
        }
        mean_numerical_pressure /= area;
        mean_analytical_pressure /= area;
    }

} // namespace Utils


#endif //MA8502_PROJECT_INTEGRATION_H
