cmake_minimum_required(VERSION 3.11)
project(TLS_EigenKokkos)

find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(RandBLAS REQUIRED)
# No need for LAPACKE or lapackpp if you are only using Eigen SVD

# Suppose you installed Eigen to $HOME/eigen
include_directories("$ENV{HOME}/eigen")

add_executable(tls_sparse_skop_kokkos_eigen tls_sparse_skop_kokkos.cc)
#add_executable(tls_dense_skop_kokkos_eigen tls_dense_skop_kokkos.cc)

target_link_libraries(tls_sparse_skop_kokkos_eigen PUBLIC
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
  RandBLAS
)

