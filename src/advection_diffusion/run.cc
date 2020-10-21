#include "rhs_ad.h"
#include "advection_diffusion.h"

int main() {
    std::cout << "PoissonNitsche" << std::endl;
    {
        double eps = 1;
        RightHandSideAD<2> rhs(eps);
        BoundaryValuesAD<2> bdd_values(eps);
        AnalyticalSolutionAD<2> exact(eps);

        AdvectionDiffusion<2> advection_diffusion(1, 7, eps, rhs, bdd_values,
                                                  exact);
        advection_diffusion.run();
    }
}