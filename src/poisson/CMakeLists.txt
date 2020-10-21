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
