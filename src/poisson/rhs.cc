//
// Created by Sigmund Eggen Holm on 17/10/2020.
//


#include "rhs.h"

#define pi 3.141592653589793

template<int dim>
double
RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    double eps = 1;
    return -eps * (-1.0 * pi * pi * sin(pi * y) * cos(pi * x) -
                   pi * pi * (1 - exp((y - 1) / eps)) * cos(pi * x) /
                   (1 - exp(-2 / eps)) - exp((y - 1) / eps) * cos(pi * x) /
                                         (eps * eps * (1 - exp(-2 / eps))));
}

template<int dim>
double
BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    double eps = 1;
    return 0.5 * sin(pi * y) * cos(pi * x) +
           (1 - exp((y - 1) / eps)) * cos(pi * x) / (1 - exp(-2 / eps));
}


template<int dim>
double AnalyticalSolution<dim>::
value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    double eps = 1;
    return 0.5 * sin(pi * y) * cos(pi * x) +
           (1 - exp((y - 1) / eps)) * cos(pi * x) / (1 - exp(-2 / eps));
}

template<int dim>
Tensor<1, dim> AnalyticalSolution<dim>::
gradient(const Point<dim> &p, const unsigned int) const {
    (void) p;
    Tensor<1, dim> value;
    double x = p[0];
    double y = p[1];
    double eps = 1;
    value[0] = -0.5 * pi * sin(pi * x) * sin(pi * y) -
               pi * (1 - exp((y - 1) / eps)) * sin(pi * x) /
               (1 - exp(-2 / eps));
    value[1] = 0.5 * pi * cos(pi * x) * cos(pi * y) -
               exp((y - 1) / eps) * cos(pi * x) / (eps * (1 - exp(-2 / eps)));
    return value;
}


template
class RightHandSide<2>;

template
class BoundaryValues<2>;

template
class AnalyticalSolution<2>;
