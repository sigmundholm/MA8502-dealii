#include <iostream>

#include "../advection_diffusion/rhs_ad.h"
#include "streamline_diffusion.h"


int main() {
    std::cout << "StreamlineDiffusion" << std::endl;
    {
        double eps = 0.1;
        RightHandSideAD<2> rhs(eps);
        BoundaryValuesAD<2> bdd_values(eps);
        AnalyticalSolutionAD<2> exact(eps);

        StreamlineDiffusion<2> streamline_diffusion(2, 5, eps,
                                                    rhs, bdd_values, exact);
        streamline_diffusion.run();
    }
}
