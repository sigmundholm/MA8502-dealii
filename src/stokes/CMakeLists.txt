add_executable(stokes run_st.cc stokes.cc rhs_st.cc ../utils/integration.h)
deal_ii_setup_target(stokes)

add_executable(convergence-st convergence_st_vec.cc stokes.cc rhs_st.cc ../utils/integration.h)
deal_ii_setup_target(convergence-st)
