#ifndef MA8502_PROJECT_RHS_VEC_H
#define MA8502_PROJECT_RHS_VEC_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;


namespace AdvectionDiffusionVector {

    struct Error {
        double mesh_size = 0;
        double l2_error = 0;
        double h1_error = 0;
    };


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
    class AnalyticalSolution : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;
    };


    template<int dim>
    class VectorField : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;
    };
} // namespace AdvectionDiffusionVector

#endif //MA8502_PROJECT_RHS_VEC_H
