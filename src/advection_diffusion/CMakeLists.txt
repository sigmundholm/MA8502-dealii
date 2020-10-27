add_executable(advection-diffusion run.cc advection_diffusion.cc rhs_ad.cc)
deal_ii_setup_target(advection-diffusion)
target_link_libraries(advection-diffusion base cutfem)

add_executable(convergence-ad convergence_ad.cc advection_diffusion.cc rhs_ad.cc)
deal_ii_setup_target(convergence-ad)
target_link_libraries(convergence-ad base cutfem)

add_executable(tut04 tut04.cc)
deal_ii_setup_target(tut04)
target_link_libraries(tut04 base cutfem)

add_executable(tut51 tut51.cc)
deal_ii_setup_target(tut51)
target_link_libraries(tut51 base cutfem)
