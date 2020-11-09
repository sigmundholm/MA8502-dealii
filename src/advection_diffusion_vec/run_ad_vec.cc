#include "rhs_ad_vec.h"
#include "advection_diffusion_vec.h"


int main() {
    using namespace AdvectionDiffusionVector;

    BoundaryValues<2> boundary_values;
    RightHandSide<2> rhs;
    AnalyticalSolution<2> analytical_solution;

    const int degree = 1;
    const int n_refines = 3;

    AdvectionDiffusion<2> advection_diffusion(degree,
                                              n_refines,
                                              rhs,
                                              boundary_values,
                                              analytical_solution,
                                              100);
    Error error = advection_diffusion.run();

    std::cout << "h = " << error.mesh_size << std::endl;
    std::cout << "e_L2 = " << error.l2_error << std::endl;
    std::cout << "e_H1 = " << error.h1_error << std::endl;
}
