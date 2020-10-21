add_executable(advection-diffusion run.cc advection_diffusion.cc rhs_ad.cc)
deal_ii_setup_target(advection-diffusion)
target_link_libraries(advection-diffusion base cutfem)
