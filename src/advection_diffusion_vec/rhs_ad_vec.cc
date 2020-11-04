

#include "rhs_ad_vec.h"

using namespace dealii;


namespace AdvectionDiffusionVector {

    template<int dim>
    double RightHandSide<dim>::point_value(const Point <dim> &p,
                                           const unsigned int) const {
        (void) p;
        return 0;
    }

    template<int dim>
    void RightHandSide<dim>::vector_value(const Point <dim> &p,
                                          Tensor<1, dim> &value) const {
        for (unsigned int c = 0; c < dim; ++c)
            value[c] = point_value(p, c);
    }

    template<int dim>
    void RightHandSide<dim>::value_list(const std::vector <Point<dim>> &points,
                                        std::vector <Tensor<1, dim>> &values) const {
        AssertDimension(points.size(), values.size());
        for (unsigned int i = 0; i < values.size(); ++i) {
            vector_value(points[i], values[i]);
        }
    }


    template<int dim>
    double BoundaryValues<dim>::point_value(const Point <dim> &p,
                                            const unsigned int component) const {
        (void) p;
        if (component == 0 && p[0] == 0) {
            if (dim == 2) {
                return -2.5 * (p[1] - 0.41) * p[1];
            }
            throw std::exception(); // TODO fix 3D
        }
        return 0;
    }

    template<int dim>
    void BoundaryValues<dim>::vector_value(const Point <dim> &p,
                                           Tensor<1, dim> &value) const {
        for (unsigned int c = 0; c < dim; ++c)
            value[c] = point_value(p, c);
    }

    template<int dim>
    void BoundaryValues<dim>::value_list(const std::vector <Point<dim>> &points,
                                         std::vector <Tensor<1, dim>> &values) const {
        AssertDimension(points.size(), values.size());
        for (unsigned int i = 0; i < values.size(); ++i) {
            vector_value(points[i], values[i]);
        }
    }


    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;
}
