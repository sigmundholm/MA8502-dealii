cmake_minimum_required(VERSION 2.8.12)

find_package(deal.II REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()

project(MA8502-project)

find_package(cutfem REQUIRED
        HINTS ${CUTFEM_DIR} $ENV{CUTFEM_DIR})

add_subdirectory(src)
