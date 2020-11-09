add_executable(stokes run_st.cc stokes.cc rhs_st.cc)
deal_ii_setup_target(stokes)