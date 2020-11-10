#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "rhs_st.h"

#define pi 3.141592653589793

using namespace dealii;

namespace Stokes {

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> val;
        val[0] = pi * sin(2 * pi * x) / 2 -
                 2 * pi * pi * sin(pi * y) * cos(pi * x);
        val[1] = 2 * pi * pi * sin(pi * x) * cos(pi * y) +
                 pi * sin(2 * pi * y) / 2;
        return val;
    }

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> val;
        val[0] = -sin(pi * y) * cos(pi * x);
        val[1] = sin(pi * x) * cos(pi * y);
        return val;
    }


    template<int dim>
    Tensor<1, dim> AnalyticalVelocity<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> val;
        val[0] = -sin(pi * y) * cos(pi * x);
        val[1] = sin(pi * x) * cos(pi * y);
        return val;
    }

    template<int dim>
    double AnalyticalPressure<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        return -cos(2 * pi * x) / 4 - cos(2 * pi * y) / 4;
    }

    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

}