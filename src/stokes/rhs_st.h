#ifndef MA8502_PROJECT_RHS_ST_H
#define MA8502_PROJECT_RHS_ST_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;

namespace Stokes {

    template<int dim>
    class RightHandSide : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;
    };


    template<int dim>
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;
    };


    template<int dim>
    class AnalyticalVelocity : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;
    };


    template<int dim>
    class AnalyticalPressure : public Function<dim> {
    public:
        double
        value(const Point<dim> &p) const override;
    };

}


#endif //MA8502_PROJECT_RHS_ST_H