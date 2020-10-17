add_library(base poisson.cc rhs.cc)
deal_ii_setup_target(base)
target_link_libraries(base cutfem)

add_executable(poisson run.cc)
deal_ii_setup_target(poisson)
target_link_libraries(poisson base cutfem)
