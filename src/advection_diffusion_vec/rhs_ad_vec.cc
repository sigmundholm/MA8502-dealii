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
        value[0] = -2 * pi * pi * sin(pi * y) * cos(pi * x);
        value[1] = 2 * pi * pi * sin(pi * x) * cos(pi * y);
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
    Tensor<1, dim> AnalyticalSolution<dim>::
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
    class AnalyticalSolution<2>;

    template
    class VectorField<2>;

} // namespace AdvectionDiffusionVector
