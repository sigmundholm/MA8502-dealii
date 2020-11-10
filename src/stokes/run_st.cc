#include <iostream>

#include "rhs_st.h"
#include "stokes.h"

int main() {
    std::cout << "Stokes" << std::endl;
    {
        const int dim = 2;
        Stokes::RightHandSide<dim> rhs;
        Stokes::BoundaryValues<dim> bdd_vals;

        Stokes::Stokes<dim> stokes(1, rhs, bdd_vals, 10);
        stokes.run();
    }
}
