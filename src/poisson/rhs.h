#ifndef MA8502_PROJECT_RHS_H
#define MA8502_PROJECT_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

using namespace dealii;


template<int dim>
class RightHandSide : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template<int dim>
class BoundaryValues : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
};


template<int dim>
class AnalyticSolution : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
};

#endif //MA8502_PROJECT_RHS_H
