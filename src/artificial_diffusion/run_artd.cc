#include "../advection_diffusion/rhs_ad.h"
#include "artificial_diffusion.h"


int main() {
    std::cout << "ArtificialDiffusion" << std::endl;
    {
        double eps = 0.1;
        RightHandSideAD<2> rhs(eps);
        BoundaryValuesAD<2> bdd_values(eps);
        AnalyticalSolutionAD<2> exact(eps);

        ArtificialDiffusion<2> artificial_diffusion(1, 5, eps, rhs, bdd_values,
                                                    exact);
        artificial_diffusion.run();
    }
}
