#include "rhs_ad.h"
#include "advection_diffusion_dg.h"

int main() {
    std::cout << "AdvectionDiffusionDG" << std::endl;
    {
        double eps = 0.1;
        RightHandSideAD<2> rhs(eps);
        BoundaryValuesAD<2> boundary_values(eps);
        AnalyticalSolutionAD<2> analytic(eps);

        AdvectionDiffusionDG<2> advec_diff(1, 5, eps, rhs, boundary_values,
                                           analytic);
        advec_diff.run();
    }
}
