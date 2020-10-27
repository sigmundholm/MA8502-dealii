add_library(base poisson.cc rhs.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(base)
target_link_libraries(base cutfem)

add_executable(poisson run.cc)
deal_ii_setup_target(poisson)
target_link_libraries(poisson base cutfem)

add_executable(error error.cc)
deal_ii_setup_target(error)
target_link_libraries(error base cutfem)

add_executable(convergence_p convergence_p.cc rhs.cc)
deal_ii_setup_target(convergence_p)
target_link_libraries(convergence_p base cutfem)

# Discontinuous Galerkin Method
add_library(base-dg poisson_dg.cc rhs.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(base-dg)
target_link_libraries(base-dg cutfem)

add_executable(poisson-dg run_dg.cc)
deal_ii_setup_target(poisson-dg)
target_link_libraries(poisson-dg base-dg base cutfem)
