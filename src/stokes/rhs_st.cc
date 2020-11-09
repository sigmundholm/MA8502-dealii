#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "rhs_st.h"

using namespace dealii;

namespace Stokes {

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point <dim> &p) const {
        Tensor<1, dim> val;
        val[0] = 0;
        val[1] = 0;
        return val;
    }

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point <dim> &p) const {
        Tensor<1, dim> val;
        val[0] = 0;
        val[1] = 0;
        if (p[0] == 0) {
            val[0] = -2.5 * (p[1] - 0.41) * p[1];
        }
        return val;
    }


    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

}