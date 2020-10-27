#include "rhs_ad.h"

#define pi 3.141592653589793


template<int dim>
RightHandSideAD<dim>::
RightHandSideAD(double eps)
        : eps(eps) {}

template<int dim>
double RightHandSideAD<dim>::
value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return 0;
}


template<int dim>
BoundaryValuesAD<dim>::
BoundaryValuesAD(double eps)
        : eps(eps) {}

template<int dim>
double BoundaryValuesAD<dim>::
value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return x * (1 - exp((y - 1) / eps)) / (1 - exp(-2 / eps));
}


template<int dim>
AnalyticalSolutionAD<dim>::
AnalyticalSolutionAD(double eps)
        : eps(eps) {}

template<int dim>
double AnalyticalSolutionAD<dim>::
value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return x * (1 - exp((y - 1) / eps)) / (1 - exp(-2 / eps));
}

template<int dim>
Tensor<1, dim> AnalyticalSolutionAD<dim>::
gradient(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    Tensor<1, dim> value;
    value[0] = (1 - exp((y - 1) / eps)) / (1 - exp(-2 / eps));
    value[1] = -x * exp((y - 1) / eps) / (eps * (1 - exp(-2 / eps)));
    return value;
}


template<int dim>
Tensor<1, dim> VectorField<dim>::
value(const Point<dim> &p) const {
    (void) p;
    Tensor<1, dim> value;
    value[0] = 0;
    value[1] = -1;  // TODO why does this work? :(
    return value;
}


template
class RightHandSideAD<2>;

template
class BoundaryValuesAD<2>;

template
class AnalyticalSolutionAD<2>;

template
class VectorField<2>;