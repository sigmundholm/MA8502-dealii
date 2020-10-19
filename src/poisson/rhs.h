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
class AnalyticalSolution : public Function<dim> {
public:
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int component = 0) const override;
};


struct Error {
    double mesh_size = 0;
    double l2_error = 0;
    double h1_error = 0;
    double h1_semi = 0;
};

#endif //MA8502_PROJECT_RHS_H
