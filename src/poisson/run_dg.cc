#include <iostream>

#include "poisson_dg.h"

using namespace dealii;


int main() {
    std::cout << "PoissonDG" << std::endl;
    {
        RightHandSide<2> rhs;
        BoundaryValues<2> boundary_values;
        AnalyticalSolution<2> analytic;

        PoissonDG<2> poisson_dg(1, 5, rhs, boundary_values, analytic);
        poisson_dg.run();
    }
}
