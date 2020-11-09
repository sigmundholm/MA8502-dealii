#include <iomanip>
#include <iostream>
#include <vector>

#include "rhs_ad_vec.h"
#include "advection_diffusion_vec.h"

using namespace AdvectionDiffusionVector;

template<int dim>
void
solve_for_element_order(const int element_order, int max_refinement) {

    std::ofstream file("errors-o" + std::to_string(element_order) + ".csv");
    AdvectionDiffusion<dim>::write_header_to_file(file);

    RightHandSide<dim> rhs;
    BoundaryValues<dim> bdd_values;
    AnalyticalSolution<dim> analytical_solution;

    for (int n_refines = 2; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;

        AdvectionDiffusion<dim> fem(element_order, n_refines,
                         rhs, bdd_values, analytical_solution);
        Error error = fem.run();
        std::cout << "|| u - u_h ||_L2 = " << error.l2_error << std::endl;
        AdvectionDiffusion<dim>::write_error_to_file(error, file);
    }
}


template<int dim>
void run_convergence_test(const std::vector<int> orders, int max_refinement) {
    for (int order : orders) {
        std::cout << "dim=" << dim << ", element_order=" << order << std::endl;
        solve_for_element_order<dim>(order, max_refinement);
    }
}


int main() {
    run_convergence_test<2>({1, 2}, 6);

}