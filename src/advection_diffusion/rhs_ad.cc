//
// Created by Sigmund Eggen Holm on 17/10/2020.
//

#include "rhs_ad.h"


template<int dim>
double
RightHandSideAD<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 1;
}


template<int dim>
double
BoundaryValuesAD<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 0;
}


template<int dim>
double
AnalyticalSolutionAD<dim>::value(const Point<dim> &p,
                                 const unsigned int) const {
    (void) p;
    return 0;
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

template class VectorField<2>;