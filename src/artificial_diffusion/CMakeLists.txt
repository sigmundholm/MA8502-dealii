add_executable(artificial-diffusion run_artd.cc artificial_diffusion.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(artificial-diffusion)
target_link_libraries(artificial-diffusion base cutfem)

add_executable(convergence-artd convergence_artd.cc artificial_diffusion.cc ../advection_diffusion/rhs_ad.cc)
deal_ii_setup_target(convergence-artd)
target_link_libraries(convergence-artd base cutfem)
