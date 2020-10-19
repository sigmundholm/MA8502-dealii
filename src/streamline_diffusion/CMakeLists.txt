add_library(sd streamline_diffusion.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(sd)
target_link_libraries(sd cutfem)

add_executable(streamline-diffusion run.cc streamline_diffusion.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(streamline-diffusion)
target_link_libraries(streamline-diffusion base sd cutfem)

add_executable(convergence convergence.cc streamline_diffusion.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(convergence)
target_link_libraries(convergence base cutfem)
