#ifndef MA8502_PROJECT_ADVECTION_DIFFUSION_DG_H
#define MA8502_PROJECT_ADVECTION_DIFFUSION_DG_H

#include "../poisson/poisson_dg.h"

template<int dim>
class AdvectionDiffusionDG : public PoissonDG<dim> {
public:
    AdvectionDiffusionDG(const unsigned int degree,
                         const unsigned int n_refines,
                         const double eps,
                         Function<dim> &rhs,
                         Function<dim> &bdd_values,
                         Function<dim> &analytical_soln);

protected:
    void assemble_system() override;

};


#endif //MA8502_PROJECT_ADVECTION_DIFFUSION_DG_H
