cmake_minimum_required(VERSION 3.11)
project(TLS_EigenKokkos)

# Find packages
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(RandBLAS REQUIRED)
find_package(lapackpp REQUIRED)  # This provides <lapack.hh>

# Eigen is header-only, assume it's installed at $HOME/eigen-3.4.x
include_directories("$ENV{HOME}/eigen-3.4.x")

# Add lapackpp include path so <lapack.hh> is found
include_directories(${lapackpp_INCLUDE_DIRS})

# Optionally, you can also add blaspp if you use <blas.hh>
# find_package(blaspp REQUIRED)
# include_directories(${blaspp_INCLUDE_DIRS})

# 1) Build the sparse TLS example
add_executable(tls_sparse_skop_kokkos_eigen tls_sparse_skop_kokkos.cc)
target_link_libraries(tls_sparse_skop_kokkos_eigen PUBLIC
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
  RandBLAS
  lapackpp
)

# 2) Build the dense TLS example
#add_executable(tls_dense_skop_kokkos_eigen tls_dense_skop_kokkos.cc)
#target_link_libraries(tls_dense_skop_kokkos_eigen PUBLIC
# Kokkos::kokkos
# KokkosKernels::kokkoskernels
# RandBLAS
# )

