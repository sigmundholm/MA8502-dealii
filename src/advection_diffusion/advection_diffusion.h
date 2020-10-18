#ifndef MA8502_PROJECT_ADVECTION_DIFFUSION_H
#define MA8502_PROJECT_ADVECTION_DIFFUSION_H

#include "../poisson/poisson.h"
#include "rhs_ad.h"

using namespace dealii;


template<int dim>
class AdvectionDiffusion : public Poisson<dim> {
public:
    AdvectionDiffusion(const unsigned int degree,
                       const unsigned int n_refines,
                       Function<dim> &rhs,
                       Function<dim> &bdd_values,
                       Function<dim> &analytical_soln);

private:
    void assemble_system() override;

    double eps = 1;
};


#endif //MA8502_PROJECT_ADVECTION_DIFFUSION_H
