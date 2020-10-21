#ifndef MA8502_PROJECT_ERROR_H
#define MA8502_PROJECT_ERROR_H

#include "poisson.h"


template<int dim>
class RightHandSidePD : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template<int dim>
class BoundaryValuesPD : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
};


template<int dim>
class AnalyticalSolutionPD : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int component = 0) const override;
};


template<int dim>
class PoissonDisk : public Poisson<dim> {
public:
    PoissonDisk(const unsigned int degree,
                const unsigned int n_refines,
                Function<dim> &rhs,
                Function<dim> &bdd_values,
                Function<dim> &analytical_soln);

private:
    void make_grid() override;
};


#endif //MA8502_PROJECT_ERROR_H
