#ifndef MA8502_PROJECT_ADVECTION_DIFFUSION_H
#define MA8502_PROJECT_ADVECTION_DIFFUSION_H

#include "../poisson/poisson.h"

using namespace dealii;


template<int dim>
class ArtificialDiffusion : public Poisson<dim> {
public:
    ArtificialDiffusion(const unsigned int degree,
                       const unsigned int n_refines,
                       const double eps,
                       Function<dim> &rhs,
                       Function<dim> &bdd_values,
                       Function<dim> &analytical_soln);

private:
    void assemble_system() override;

};


#endif //MA8502_PROJECT_ADVECTION_DIFFUSION_H
