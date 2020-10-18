add_executable(streamline-diffusion streamline_diffusion.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(streamline-diffusion)
target_link_libraries(streamline-diffusion base cutfem)
