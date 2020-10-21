#include <iostream>

#include "poisson.h"

using namespace dealii;


int main() {
    std::cout << "PoissonNitsche" << std::endl;
    {
        RightHandSide<2> rhs;
        BoundaryValues<2> boundary_values;
        AnalyticalSolution<2> analytic;

        Poisson<2> poisson(1, 5, rhs, boundary_values, analytic);
        poisson.run();
    }
}
