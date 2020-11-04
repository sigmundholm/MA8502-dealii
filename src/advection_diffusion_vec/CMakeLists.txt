add_executable(advection-diffusion-vec run_ad_vec.cc advection_diffusion_vec.cc rhs_ad_vec.cc)
deal_ii_setup_target(advection-diffusion-vec)
