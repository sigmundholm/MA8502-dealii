//
// Created by Sigmund Eggen Holm on 17/10/2020.
//

#include "rhs_ad.h"

#define pi 3.14159265
#define eps 1

template<int dim>
double
RightHandSideAD<dim>::value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return 0;
}


template<int dim>
double
BoundaryValuesAD<dim>::value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return x * (1 - exp((y - 1) / eps)) / (1 - exp(-2 / eps));
}


template<int dim>
double
AnalyticalSolutionAD<dim>::value(const Point<dim> &p,
                                 const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return x * (1 - exp((y - 1) / eps)) / (1 - exp(-2 / eps));
}


template<int dim>
Tensor<1, dim> VectorField<dim>::
value(const Point<dim> &p) const {
    (void) p;
    Tensor<1, dim> value;
    value[0] = 0;
    value[1] = 1;
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