#include "deal.II/grid/grid_generator.h"
#include "deal.II/base/point.h"


#include "error.h"

#define pi 3.141592653589793

template<int dim>
double
RightHandSidePD<dim>::value(const Point<dim> &p, const unsigned int) const {
    double r_sq = p[0] * p[0] + p[1] * p[1];
    return -8 * pi * cos(2 * pi * r_sq)
           + 16 * pi * pi * r_sq * sin(2 * pi * r_sq);
}

template<int dim>
double
BoundaryValuesPD<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 0;
}


template<int dim>
double AnalyticalSolutionPD<dim>::
value(const Point<dim> &p, const unsigned int) const {
    double r_sq = p[0] * p[0] + p[1] * p[1];
    return sin(2 * pi * r_sq);
}

template<int dim>
Tensor<1, dim> AnalyticalSolutionPD<dim>::
gradient(const Point<dim> &p, const unsigned int) const {
    (void) p;
    Tensor<1, dim> value;

    double r_sq = p[0] * p[0] + p[1] * p[1];
    value[0] = 4 * pi * p[0] * cos(2 * pi * r_sq);
    value[1] = 4 * pi * p[1] * cos(2 * pi * r_sq);
    return value;
}


template<int dim>
PoissonDisk<dim>::
PoissonDisk(const unsigned int degree,
            const unsigned int n_refines,
            Function<dim> &rhs,
            Function<dim> &bdd_values,
            Function<dim> &analytical_soln)
        : Poisson<dim>(degree, n_refines, rhs, bdd_values, analytical_soln) {

}

template<int dim>
void PoissonDisk<dim>::make_grid() {
    // GridGenerator::cylinder(triangulation, 5, 10);
    Point<dim> center(0, 0);
    GridGenerator::hyper_ball(this->triangulation, center);
    this->triangulation.refine_global(this->n_refines);

    std::cout << "  Number of active cells: "
              << this->triangulation.n_active_cells()
              << std::endl;
}


int main() {
    std::cout << "PoissonDisk" << std::endl;

    RightHandSidePD<2> rhs;
    BoundaryValuesPD<2> bdd;
    AnalyticalSolutionPD<2> as;

    PoissonDisk<2> poisson(2, 5, rhs, bdd, as);
    Error error = poisson.run();
}