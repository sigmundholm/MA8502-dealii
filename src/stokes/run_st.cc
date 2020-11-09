#include <iostream>
#include "stokes.h"

int main() {
    std::cout << "Stokes" << std::endl;
    {
        using namespace Stokes;

        const int dim = 2;
        RightHandSide<dim> rhs;
        BoundaryValues<dim> bdd_vals;

        StokesNitsche<dim> stokesNitsche(1, rhs, bdd_vals);
        stokesNitsche.run();
    }
}
