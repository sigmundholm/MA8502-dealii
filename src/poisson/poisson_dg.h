#ifndef MA8502_PROJECT_POISSON_DG_H
#define MA8502_PROJECT_POISSON_DG_H

#include "deal.II/fe/fe_dgq.h"

#include "poisson.h"


template<int dim>
class PoissonDG : public Poisson<dim> {
public:
    PoissonDG(const unsigned int degree,
              const unsigned int n_refines,
              Function<dim> &rhs,
              Function<dim> &bdd_values,
              Function<dim> &analytical_soln);

protected:
    void setup_system() override;

    void assemble_system() override;

    FE_DGQ<dim> fe;
};


#endif //MA8502_PROJECT_POISSON_DG_H
