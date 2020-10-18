add_executable(artificial-diffusion artificial_diffusion.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(artificial-diffusion)
target_link_libraries(artificial-diffusion base cutfem)
