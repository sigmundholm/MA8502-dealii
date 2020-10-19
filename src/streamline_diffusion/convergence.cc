#include <iomanip>
#include <iostream>
#include <vector>

#include "../advection_diffusion/rhs_ad.h"
#include "streamline_diffusion.h"

template<int dim>
void
solve_for_element_order(const int element_order, int max_refinement,
                        double eps) {

    std::ofstream file("errors-o" + std::to_string(element_order)
                       + "-eps=" + std::to_string(eps) + ".csv");
    StreamlineDiffusion<dim>::write_header_to_file(file);

    RightHandSideAD<dim> rhs(eps);
    BoundaryValuesAD<dim> bdd_values(eps);
    AnalyticalSolutionAD<dim> analytical_solution(eps);

    for (int n_refines = 2; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;

        StreamlineDiffusion<dim> fem(element_order, n_refines, eps,
                                     rhs, bdd_values, analytical_solution);
        Error error = fem.run();
        std::cout << "|| u - u_h ||_L2 = " << error.l2_error << std::endl;
        StreamlineDiffusion<dim>::write_error_to_file(error, file);
    }
}


template<int dim>
void run_convergence_test(const std::vector<int> orders, int max_refinement,
                          double eps) {
    for (int order : orders) {
        std::cout << "dim=" << dim << ", element_order=" << order << std::endl;
        solve_for_element_order<dim>(order, max_refinement, eps);
    }
}


int main() {
    double eps = 0.1;
    run_convergence_test<2>({2}, 6, eps);

}