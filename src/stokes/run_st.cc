#include <iostream>

#include "rhs_st.h"
#include "stokes.h"

int main() {
    std::cout << "Stokes" << std::endl;
    {
        const int dim = 2;
        const int n_refines = 2;
        Stokes::RightHandSide <dim> rhs;
        Stokes::BoundaryValues <dim> bdd_vals;
        Stokes::AnalyticalVelocity <dim> analytical_u;
        Stokes::AnalyticalPressure <dim> analytical_p;

        Stokes::Stokes <dim> stokes(1, n_refines, rhs, bdd_vals, analytical_u,
                                    analytical_p, 10);
        Stokes::Error error = stokes.run();
        std::cout << "u_L2 = " << error.u_l2_error << std::endl;
        std::cout << "u_H1-semi = " << error.u_h1_semi_error << std::endl;
        std::cout << "u_H1 = " << error.u_h1_error << std::endl;
        std::cout << "p_L2 = " << error.p_l2_error << std::endl;
        std::cout << "p_H1-semi = " << error.p_h1_semi_error << std::endl;
        std::cout << "p_H1 = " << error.p_h1_error << std::endl;
    }
}
