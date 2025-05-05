cmake_minimum_required(VERSION 3.10)

project(examples)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/kokkos-install")
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O1")

find_package(OpenMP REQUIRED)

message(STATUS "Checking for RandBLAS ... ")
find_package(RandBLAS REQUIRED)
message(STATUS "Done checking for RandBLAS. ...")

message(STATUS "Looking for LAPACK++ ... ")
find_package(lapackpp REQUIRED)
message(STATUS "Done looking for LAPACK++.")

# Find Random123
find_path(Random123_INCLUDE_DIR Random123/philox.h
    PATHS
    $ENV{HOME}/rsh/random123-install/include
    $ENV{HOME}/random123-install/include
    /usr/local/include
    /usr/include
)

if(Random123_INCLUDE_DIR)
    message(STATUS "Found Random123 headers at: ${Random123_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Random123 headers not found")
endif()

# Find Eigen
find_path(Eigen3_INCLUDE_DIR Eigen/Dense
    PATHS
    $ENV{HOME}/rsh/eigen-install/include/eigen3
    $ENV{HOME}/eigen-master
    /usr/local/include/eigen3
    /usr/include/eigen3
)

if(Eigen3_INCLUDE_DIR)
    message(STATUS "Found Eigen headers at: ${Eigen3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen headers not found")
endif()

set(
    tls_dense_skop_cxx total-least-squares/tls_dense_skop.cc
)
add_executable(
    tls_dense_skop ${tls_dense_skop_cxx}
)
target_include_directories(
    tls_dense_skop PUBLIC ${Random123_INCLUDE_DIR}
)
target_link_libraries(
    tls_dense_skop PUBLIC RandBLAS blaspp lapackpp 
)

set(
    tls_sparse_skop_cxx total-least-squares/tls_sparse_skop.cc
)
add_executable(
    tls_sparse_skop ${tls_sparse_skop_cxx}
)
target_include_directories(
    tls_sparse_skop PUBLIC ${Random123_INCLUDE_DIR}
)
target_link_libraries(
    tls_sparse_skop PUBLIC RandBLAS blaspp lapackpp 
)

set(
    tls_sparse_skop_kokkos_cxx total-least-squares/tls_sparse_skop_kokkos.cc
)
add_executable(
    tls_sparse_skop_kokkos ${tls_sparse_skop_kokkos_cxx}
)
target_link_libraries(
    tls_sparse_skop_kokkos PUBLIC Kokkos::kokkos KokkosKernels::kokkoskernels RandBLAS blaspp lapackpp
)

add_executable(
    slra_svd_synthetic sparse-low-rank-approx/svd_rank1_plus_noise.cc
)
target_include_directories(
    slra_svd_synthetic PUBLIC ${Random123_INCLUDE_DIR}
)
target_link_libraries(
    slra_svd_synthetic PUBLIC RandBLAS blaspp lapackpp
)


include(FetchContent)
FetchContent_Declare(
    fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(
    fast_matrix_market
)

add_executable(
    slra_svd_fmm sparse-low-rank-approx/svd_matrixmarket.cc
)
target_include_directories(
    slra_svd_fmm PUBLIC ${Random123_INCLUDE_DIR}
)
target_link_libraries(
    slra_svd_fmm PUBLIC RandBLAS blaspp lapackpp fast_matrix_market::fast_matrix_market
)

add_executable(
    slra_qrcp sparse-low-rank-approx/qrcp_matrixmarket.cc
)
target_include_directories(
    slra_qrcp PUBLIC ${Random123_INCLUDE_DIR}
)
target_link_libraries(
    slra_qrcp PUBLIC RandBLAS blaspp lapackpp fast_matrix_market::fast_matrix_market
)

set(
    tls_sparse_skop_philox_cxx total-least-squares/tls_sparse_skop_philox.cc
)
add_executable(
    tls_sparse_skop_philox ${tls_sparse_skop_philox_cxx}
)
# Add the missing include directory for tls_sparse_skop_philox
target_include_directories(
    tls_sparse_skop_philox PUBLIC ${Random123_INCLUDE_DIR}
)
target_link_libraries(
    tls_sparse_skop_philox PUBLIC Kokkos::kokkos KokkosKernels::kokkoskernels RandBLAS blaspp lapackpp
)

set(
    tls_dense_skop_kokkos_cxx total-least-squares/tls_dense_skop_kokkos.cc
)
add_executable(
    tls_dense_skop_kokkos ${tls_dense_skop_kokkos_cxx}
)
target_include_directories(
    tls_dense_skop_kokkos PUBLIC ${Random123_INCLUDE_DIR}
)
target_include_directories(
    tls_dense_skop_kokkos PUBLIC ${Eigen3_INCLUDE_DIR}
)
target_link_libraries(
    tls_dense_skop_kokkos PUBLIC Kokkos::kokkos KokkosKernels::kokkoskernels RandBLAS blaspp lapackpp
)
