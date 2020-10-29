add_executable(advection-diffusion run.cc advection_diffusion.cc rhs_ad.cc)
deal_ii_setup_target(advection-diffusion)
target_link_libraries(advection-diffusion base cutfem)

add_executable(convergence-ad convergence_ad.cc advection_diffusion.cc rhs_ad.cc)
deal_ii_setup_target(convergence-ad)
target_link_libraries(convergence-ad base cutfem)

add_executable(advection-diffusion-dg run_dg.cc advection_diffusion_dg.cc rhs_ad.cc)
deal_ii_setup_target(advection-diffusion-dg)
target_link_libraries(advection-diffusion-dg base-dg base cutfem)

add_executable(convergence-dg convergence_dg.cc advection_diffusion_dg.cc rhs_ad.cc)
deal_ii_setup_target(convergence-dg)
target_link_libraries(convergence-dg base-dg base cutfem)
