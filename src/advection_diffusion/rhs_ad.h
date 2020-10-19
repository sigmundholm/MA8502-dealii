#ifndef MA8502_PROJECT_RHS_H
#define MA8502_PROJECT_RHS_H

#include "deal.II/base/function.h"
#include "deal.II/base/point.h"
#include "deal.II/base/tensor_function.h"


using namespace dealii;


template<int dim>
class RightHandSideAD : public Function<dim> {
public:
    RightHandSideAD(double eps);

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
    const double eps;
};


template<int dim>
class BoundaryValuesAD : public Function<dim> {
public:
    BoundaryValuesAD(double eps);

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
    const double eps;
};


template<int dim>
class AnalyticalSolutionAD : public Function<dim> {
public:
    AnalyticalSolutionAD(double eps);

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int component = 0) const override;

private:
    const double eps;
};


template<int dim>
class VectorField : public TensorFunction<1, dim> {
public:
    Tensor<1, dim> value(const Point<dim> &p) const override;
};


#endif //MA8502_PROJECT_RHS_H
