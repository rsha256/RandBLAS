/********************************************************************************
 * tls_dense_skop_kokkos_eigen.cpp
 *
 * A standalone example of Total Least Squares using Kokkos for data
 * initialization, plus Eigen for the SVD (no LAPACKE dependency).
 *
 * NOTE:
 * If you get warnings about calling constexpr __host__ functions from
 * __host__ __device__ functions in Eigen, compile with:
 *    -DCMAKE_CXX_FLAGS="-expt-relaxed-constexpr"
 * (only needed if building with NVCC/CUDA).
 ********************************************************************************/

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cstdlib>

KOKKOS_INLINE_FUNCTION
double box_muller(double u1, double u2) {
  double r = std::sqrt(-2.0 * std::log(u1));
  double theta = 2.0 * M_PI * u2;
  return r * std::cos(theta);
}

// Initialize [A|B] in AB (size m x (n+1)):
// - columns 0..(n-1) store A
// - column n stores B
// B = A*ones + gaussian noise
void init_noisy_data(int64_t m, int64_t n, Kokkos::View<double**> AB) {
  using execution_space = Kokkos::DefaultExecutionSpace;

  Kokkos::View<double*> target_x("target_x", n);
  Kokkos::View<double*> eps("eps", m);

  // Random pool
  Kokkos::Random_XorShift64_Pool<execution_space> pool(0);

  // target_x(i) = 1
  Kokkos::parallel_for("FillTargetX", n, KOKKOS_LAMBDA(const int i) {
    target_x(i) = 1.0;
  });

  // Fill A with normal random data
  Kokkos::parallel_for("FillA", m*n, KOKKOS_LAMBDA(const int idx) {
    int i = idx % m;
    int j = idx / m;
    auto rand_gen = pool.get_state();
    double u1 = Kokkos::rand<decltype(rand_gen), double>::draw(rand_gen);
    double u2 = Kokkos::rand<decltype(rand_gen), double>::draw(rand_gen);
    pool.free_state(rand_gen);
    AB(i, j) = box_muller(u1, u2);
  });

  // Fill eps with normal random data
  Kokkos::parallel_for("FillEps", m, KOKKOS_LAMBDA(const int i) {
    auto rand_gen = pool.get_state();
    double u1 = Kokkos::rand<decltype(rand_gen), double>::draw(rand_gen);
    double u2 = Kokkos::rand<decltype(rand_gen), double>::draw(rand_gen);
    pool.free_state(rand_gen);
    eps(i) = box_muller(u1, u2);
  });

  // B = A * target_x + eps
  {
    Kokkos::View<double*> B("B", m);
    auto A_sub = Kokkos::subview(AB, Kokkos::ALL(), Kokkos::make_pair((size_t)0, (size_t)n));
    KokkosBlas::gemv("N", 1.0, A_sub, target_x, 0.0, B);

    Kokkos::parallel_for("AddEps", m, KOKKOS_LAMBDA(const int i) {
      B(i) += eps(i);
    });

    // Copy B into AB(:, n)
    Kokkos::parallel_for("StoreB", m, KOKKOS_LAMBDA(const int i) {
      AB(i, n) = B(i);
    });
  }
}

// Perform Total Least Squares using Eigen SVD on host:
template <typename T>
void total_least_squares_eigen(int64_t m, int64_t n,
                               Kokkos::View<T**> AB_d,  // device memory
                               Kokkos::View<T*>   x_d,
                               Kokkos::View<T*>   svals_d,
                               Kokkos::View<T**>  VT_d)
{
  // 1) Create host mirror of AB
  auto AB_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), AB_d);

  // 2) Map that memory into an Eigen matrix (m x (n+1), col-major)
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
    A_eig(AB_h.data(), m, n+1);

  // 3) SVD using Eigen's JacobiSVD: get V fully
  Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> svd(
      A_eig, Eigen::ComputeFullV);
  auto sigma = svd.singularValues();  // size = n+1
  auto Vmat  = svd.matrixV();         // (n+1) x (n+1)

  // 4) Copy SVD results into host arrays
  // -- 4a) singular values
  {
    // rank-1 deep_copy from host to device is always fine,
    // because rank-1 layout is effectively the same.
    Kokkos::View<T*, Kokkos::HostSpace> svals_h("svals_h", n+1);
    for (int i = 0; i < n+1; i++)
      svals_h(i) = sigma(i);
    Kokkos::deep_copy(svals_d, svals_h);
  }

  // -- 4b) solution x: dimension n
  {
    Kokkos::View<T*, Kokkos::HostSpace> x_h("x_h", n);
    double scale = Vmat(n, n);
    for (int i = 0; i < n; i++) {
      x_h(i) = -Vmat(n, i) / scale;
    }
    Kokkos::deep_copy(x_d, x_h);
  }

  // -- 4c) V itself, for debugging or analysis
  // Instead of creating a brand-new Kokkos::View with default layout,
  // create a mirror of the device View so that layout matches what
  // device expects, and the subsequent deep_copy is valid.
  auto VT_h = Kokkos::create_mirror_view(VT_d);

  // Fill row i, column j of host mirror
  for (int i = 0; i < (int)n+1; i++) {
    for (int j = 0; j < (int)n+1; j++) {
      // Vmat is col-major in Eigen, so Vmat(i,j) is i-th row, j-th col
      // but in memory we are doing (i, j) => up to you if you want to
      // transpose or keep consistent with your usage
      VT_h(i, j) = Vmat(i, j);
    }
  }

  // 5) Copy from host mirror to device
  Kokkos::deep_copy(VT_d, VT_h);
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    int64_t m, n;
    if (argc == 1) {
      m = 10000;
      n = 500;
    } else if (argc == 3) {
      m = std::atoll(argv[1]);
      n = std::atoll(argv[2]);
      if (n > m) {
        std::cout << "Make sure m >= n\n";
        Kokkos::finalize();
        return 0;
      }
    } else {
      std::cout << "Usage: ./tls_dense_skop_kokkos_eigen <m> <n>\n";
      Kokkos::finalize();
      return 1;
    }

    // Embedding dimension for the random S
    int64_t sk_dim = 2*(n+1);

    // Create device Views
    Kokkos::View<double**> AB("AB", m, n+1);
    Kokkos::View<double**> SAB("SAB", sk_dim, n+1);
    Kokkos::View<double*>  sketch_x("sketch_x", n);
    Kokkos::View<double*>  svals("svals", n+1);
    Kokkos::View<double**> VT("VT", n+1, n+1);

    // 1) init noisy data
    init_noisy_data(m, n, AB);
    std::cout << "\n[A|B]: " << m << " x " << (n+1)
              << "\nEmbedding dimension: " << sk_dim << "\n";

    // 2) Create random S (sk_dim x m), compute SAB = S * AB
    auto t0 = std::chrono::high_resolution_clock::now();
    Kokkos::View<double**> S("S", sk_dim, m);
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(1997);
    // FillS
    Kokkos::parallel_for("FillS", sk_dim*m, KOKKOS_LAMBDA(const int idx){
      int i = idx % sk_dim;
      int j = idx / sk_dim;
      auto rand_gen = pool.get_state();
      double u1 = Kokkos::rand<decltype(rand_gen), double>::draw(rand_gen);
      double u2 = Kokkos::rand<decltype(rand_gen), double>::draw(rand_gen);
      pool.free_state(rand_gen);
      S(i, j) = box_muller(u1, u2);
    });
    auto t1 = std::chrono::high_resolution_clock::now();
    double sample_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nTime to sample S: " << sample_time << " s\n";

    // gemm: SAB = S(sk_dim x m) * AB(m x (n+1)) => (sk_dim x (n+1))
    auto t2 = std::chrono::high_resolution_clock::now();
    KokkosBlas::gemm("N", "N", 1.0, S, AB, 0.0, SAB);
    auto t3 = std::chrono::high_resolution_clock::now();
    double sketch_time = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "Time to compute SAB=S*AB: " << sketch_time << " s\n";

    // 3) Randomized TLS
    auto t4 = std::chrono::high_resolution_clock::now();
    total_least_squares_eigen(sk_dim, n, SAB, sketch_x, svals, VT);
    auto t5 = std::chrono::high_resolution_clock::now();
    double randomized_time = std::chrono::duration<double>(t5 - t4).count();
    std::cout << "Randomized TLS time: " << randomized_time << " s\n";

    // 4) Classical TLS on full AB
    Kokkos::View<double*> true_x("true_x", n);
    auto t6 = std::chrono::high_resolution_clock::now();
    total_least_squares_eigen(m, n, AB, true_x, svals, VT);
    auto t7 = std::chrono::high_resolution_clock::now();
    double classical_time = std::chrono::duration<double>(t7 - t6).count();
    std::cout << "Classical TLS time: " << classical_time << " s\n";
    std::cout << "Speedup: " << (classical_time / randomized_time) << "\n";

    // 5) Compare solutions: delta = sketch_x - true_x
    Kokkos::View<double*> delta("delta", n);
    // y = alpha*x + beta*y => delta = 1.0*sketch_x + 0.0*delta
    KokkosBlas::axpby(1.0, sketch_x, 0.0, delta);
    // delta = -1.0*true_x + 1.0*delta => delta -= true_x
    KokkosBlas::axpby(-1.0, true_x, 1.0, delta);

    // nrm2 outputs to a 0D device view
    Kokkos::View<double> dev_nrm_delta("dev_nrm_delta");
    Kokkos::View<double> dev_nrm_true("dev_nrm_true");
    KokkosBlas::nrm2(dev_nrm_delta, delta);
    KokkosBlas::nrm2(dev_nrm_true,  true_x);

    double h_delta=0.0, h_true=0.0;
    Kokkos::deep_copy(h_delta, dev_nrm_delta);
    Kokkos::deep_copy(h_true,  dev_nrm_true);

    double rel_err = h_delta / h_true;
    std::cout << "Relative error: " << rel_err << "\n";
  }
  Kokkos::finalize();
  return 0;
}

