#ifndef MA8502_PROJECT_STREAMLINE_DIFFUSION_H
#define MA8502_PROJECT_STREAMLINE_DIFFUSION_H


#include "../poisson/poisson.h"

using namespace dealii;


template<int dim>
class StreamlineDiffusion : public Poisson<dim> {
public:
    StreamlineDiffusion(const unsigned int degree,
                        const unsigned int n_refines,
                        const double eps,
                        Function<dim> &rhs,
                        Function<dim> &bdd_values,
                        Function<dim> &analytical_soln);

private:
    void assemble_system() override;

    // double eps = 1;
};



#endif //MA8502_PROJECT_STREAMLINE_DIFFUSION_H
