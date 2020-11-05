#include "rhs_ad_vec.h"
#include "advection_diffusion_vec.h"


int main() {
    using namespace AdvectionDiffusionVector;

    BoundaryValues<2> boundary_values;
    RightHandSide<2> rhs;

    const int degree = 1;


    StokesNitsche<2> stokes(degree,
                           rhs,
                           boundary_values,
                           100);
    stokes.run();
}
