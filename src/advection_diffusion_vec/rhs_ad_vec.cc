#include "rhs_ad_vec.h"

#define pi 3.141592653589793

using namespace dealii;


namespace AdvectionDiffusionVector {

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];

        Tensor<1, dim> value;
        value[0] = -pi * sin(pi * x) * sin(pi * y) * sin(pi * y) * cos(pi * x) -
                   pi * sin(pi * x) * cos(pi * x) * cos(pi * y) * cos(pi * y) -
                   2 * pi * pi * sin(pi * y) * cos(pi * x);
        value[1] = -pi * sin(pi * x) * sin(pi * x) * sin(pi * y) * cos(pi * y) +
                   2 * pi * pi * sin(pi * x) * cos(pi * y) -
                   pi * sin(pi * y) * cos(pi * x) * cos(pi * x) * cos(pi * y);
        return value;
    }

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> value;
        value[0] = -sin(pi * y) * cos(pi * x);
        value[1] = sin(pi * x) * cos(pi * y);
        return value;
    }


    template<int dim>
    Tensor<1, dim> VectorField<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> value;
        value[0] = -sin(pi * y) * cos(pi * x);
        value[1] = sin(pi * x) * cos(pi * y);
        return value;
    }


    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

    template
    class VectorField<2>;
}
