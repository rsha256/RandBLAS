#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>

// Add philox header from Random123
#include <Random123/philox.h>
#include <Random123/uniform.hpp>

// BLAS++ / LAPACK++ headers for lapack::gesdd, etc.
#include <blas.hh>
#include <lapack.hh>

// Type definitions for philox
typedef r123::Philox4x32 PHILOX;
typedef PHILOX::ctr_type PhiloxCounterType;
typedef PHILOX::key_type PhiloxKeyType;

//---------------------------------------------
// Fill [A|B] in parallel with random data, then
//   B = A * ones + Gaussian noise
//---------------------------------------------
KOKKOS_INLINE_FUNCTION
double box_muller(const double u1, const double u2) {
  // Convert uniform random u1,u2 in (0,1) -> normal(0,1)
  double r     = sqrt(-2.0 * log(u1));
  double theta = 2.0 * M_PI * u2;
  return r * cos(theta);
}

void init_noisy_data_kokkos(const int64_t m,
                            const int64_t n,
                            Kokkos::View<double**> AB)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  // Fill columns 0..(n-1) of AB (the "A" part) with normal random data
  Kokkos::parallel_for("FillA", 
    Kokkos::RangePolicy<ExecSpace>(0, m*n),
    KOKKOS_LAMBDA(const int idx)
  {
    int i = idx % m;       // row
    int j = idx / m;       // col in [0..n-1]
    
    // Using philox to generate random numbers
    PhiloxCounterType ctr = {{(uint32_t)idx, 0, 0, 0}};
    PhiloxKeyType key = {{12345, 67890}};
    PHILOX philox;
    
    PhiloxCounterType rand = philox(ctr, key);
    // Convert to uniform [0,1) using r123's u01 function
    double u1 = r123::u01<double>(rand[0]); // First random number
    
    // For the second random number, increment counter
    ctr.incr();
    rand = philox(ctr, key);
    double u2 = r123::u01<double>(rand[0]);
    
    AB(i, j) = box_muller(u1, u2);  // normal(0,1)
  });

  // 3) B = A*ones + noise
  //    We can do that with: B(i) = sum_j A(i,j) + noise
  //    We'll form the noise below, then add the matrix-vector product.

  // Temporary device vector for eps
  Kokkos::View<double*> eps("eps", m);
  // Fill eps with normal(0,1)
  Kokkos::parallel_for("FillEps", 
    Kokkos::RangePolicy<ExecSpace>(0, m),
    KOKKOS_LAMBDA(const int i)
  {
    // Using philox for noise generation
    PhiloxCounterType ctr = {{(uint32_t)i, 123, 456, 0}}; // Different starting sequence
    PhiloxKeyType key = {{12345, 67890}};
    PHILOX philox;
    
    PhiloxCounterType rand = philox(ctr, key);
    double u1 = r123::u01<double>(rand[0]);
    
    ctr.incr();
    rand = philox(ctr, key);
    double u2 = r123::u01<double>(rand[0]);

    eps(i) = box_muller(u1, u2);
  });

  // 4) B = A*(ones) + eps
  //    The last column in AB is "B".
  //    Use gemv for A*(1,1,...,1), then add eps.
  {
    Kokkos::View<double*> ones("ones", n);
    // Fill "ones" with 1.0
    Kokkos::deep_copy(ones, 1.0);

    auto A_sub = Kokkos::subview(AB, Kokkos::ALL(), 
                                 Kokkos::make_pair((size_t)0, (size_t)n));
    Kokkos::View<double*> B("B", m);

    // B = A*ones
    KokkosBlas::gemv("N", 1.0, A_sub, ones, 0.0, B);

    // B(i) += eps(i)
    Kokkos::parallel_for("AddNoise",
      Kokkos::RangePolicy<ExecSpace>(0, m),
      KOKKOS_LAMBDA(const int i)
    {
      B(i) += eps(i);
    });

    // Copy B back into AB(:, n)
    Kokkos::parallel_for("StoreB",
      Kokkos::RangePolicy<ExecSpace>(0, m),
      KOKKOS_LAMBDA(const int i)
    {
      AB(i, n) = B(i);
    });
  }
}

//---------------------------------------------
// Build a "sparse" random sketch S with 8 nnz/col
// S stored in a compressed form: for each column j,
// we store rowIndices(8*j..8*j+7) and values(8*j..8*j+7).
//---------------------------------------------
void init_sparse_sk_kokkos(const int64_t m,        // # columns in S
                           const int64_t sk_dim,   // # rows in S
                           const int64_t nnz_per_col,
                           Kokkos::View<int*> rowIdx,
                           Kokkos::View<double*> vals)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  Kokkos::parallel_for("SparseS_init",
    Kokkos::RangePolicy<ExecSpace>(0, m),
    KOKKOS_LAMBDA(const int j)
  {
    // For each column j, pick nnz_per_col random row positions and random values.
    // We'll store them in contiguous slices: rowIdx( nnz_per_col*j + i_nz ), ...
    
    for (int i_nz = 0; i_nz < nnz_per_col; i_nz++) {
      // Set up philox with unique counter-key combination for each entry
      PhiloxCounterType ctr = {{(uint32_t)j, (uint32_t)i_nz, 789, 0}};
      PhiloxKeyType key = {{9999, 8888}};
      PHILOX philox;
      
      // Generate random row index
      PhiloxCounterType rand = philox(ctr, key);
      // Convert to integer in range [0, sk_dim)
      int r = (int)(r123::u01<double>(rand[0]) * sk_dim);
      if (r >= sk_dim) r = sk_dim - 1; // Safety check
      
      // Generate random normal value
      ctr.incr();
      rand = philox(ctr, key);
      double u1 = r123::u01<double>(rand[0]);
      
      ctr.incr();
      rand = philox(ctr, key);
      double u2 = r123::u01<double>(rand[0]);
      
      double val = box_muller(u1, u2);

      rowIdx(nnz_per_col*j + i_nz) = r;
      vals(nnz_per_col*j + i_nz)   = val;
    }
  });
}

//---------------------------------------------
// SPMM:  SAB = S x AB
//   S is stored in compressed form: rowIdx, vals
//   For each column j of AB, and each k in [0..nABcols),
//   we add the contribution to SAB(row, k).
//   We do atomic adds since multiple threads might update SAB(row,k).
//---------------------------------------------
void spmm_sparse_kokkos(const int64_t m, 
                        const int64_t sk_dim,
                        const int64_t nnz_per_col,
                        Kokkos::View<int*> rowIdx,
                        Kokkos::View<double*> vals,
                        Kokkos::View<double**> AB,
                        Kokkos::View<double**> SAB)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  const int64_t nABcols = SAB.extent(1);  // # of columns in AB => same as # columns in SAB

  // We assume SAB is initialized to zero before this call.
  Kokkos::deep_copy(SAB, 0.0);

  // 2D range over (j in [0..m), k in [0..nABcols))
  Kokkos::parallel_for("spmm",
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>(
      {0, 0}, {m, nABcols}),
    KOKKOS_LAMBDA(const int j, const int k)
  {
    // Retrieve A(j,k)
    double a_jk = AB(j, k);

    // For each nnz in column j
    for (int i_nz = 0; i_nz < nnz_per_col; i_nz++) {
      int i = rowIdx(nnz_per_col*j + i_nz);
      double c = vals(nnz_per_col*j + i_nz);

      // Atomic add to SAB(i,k) with c*A(j,k)
      Kokkos::atomic_add(&SAB(i,k), c * a_jk);
    }
  });
}

//---------------------------------------------
// TLS via lapack::gesdd on Host
// Overwrites AB_h with U, returns S in svals_h, V^T in VT_h,
// and the final solution vector in x_h
//---------------------------------------------
void total_least_squares_cpu(const int64_t m, const int64_t n,
                             Kokkos::View<double**>  AB_d,
                             Kokkos::View<double*>   x_d)
{
  // Copy device -> host
  auto AB_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), AB_d);

  // We'll use a local host workspace for SVD
  std::vector<double> svals(n+1);
  std::vector<double> VT((n+1)*(n+1));

  // Perform SVD with lapack::gesdd in Overwrite mode
  //   AB_h is m x (n+1) in col-major => leading dimension m
  lapack::gesdd(
    lapack::Job::OverwriteVec, // overwrite AB_h
    m, n+1,
    AB_h.data(), m,            // A->U in Overwrite
    svals.data(),
    nullptr, 1,                // U is overwritten in AB_h, we don't need it
    VT.data(), n+1             // V^T
  );

  // Last entry of V is V((n+1)-1, (n+1)-1) => scale
  double scale = VT[(n+1)*(n+1)-1]; // bottom-right corner
  // x(i) = - V(n, i) / scale  for i=0..n-1
  // but remember V^T is stored in "VT", so V(n, i) is VT(i, n).
  std::vector<double> x_h(n);
  for (int i = 0; i < n; i++) {
    x_h[i] = -VT[i*(n+1) + n] / scale;
  }

  // Copy x_h -> device x_d
  Kokkos::View<double*, Kokkos::HostSpace> x_temp("x_temp", n);
  for (int i = 0; i < n; i++)
    x_temp(i) = x_h[i];

  Kokkos::deep_copy(x_d, x_temp);
}

//---------------------------------------------
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    // 1) Parse arguments
    int64_t m = 10000, n = 500;
    if (argc == 3) {
      m = std::atoll(argv[1]);
      n = std::atoll(argv[2]);
      if (n > m) {
        std::cout << "Make sure m >= n\n";
        Kokkos::finalize();
        return 0;
      }
    } else if (argc != 1) {
      std::cout << "Usage: ./executable <m> <n>\n";
      Kokkos::finalize();
      return 1;
    }

    // 2) Dimensions
    int64_t sk_dim = 2*(n+1);  // row dimension of sketch
    std::cout << "\n[A|B]: " << m << " x " << (n+1)
              << "\nEmbedding dimension = " << sk_dim << std::endl;

    // 3) Allocate device memory for AB, SAB, solutions
    Kokkos::View<double**> AB ("AB",  m, n+1);
    Kokkos::View<double**> SAB("SAB", sk_dim, n+1);

    Kokkos::View<double*>  sketch_x("sketch_x", n);
    Kokkos::View<double*>  true_x  ("true_x",   n);

    // 4) Initialize [A|B] in parallel
    auto t0 = std::chrono::high_resolution_clock::now();
    init_noisy_data_kokkos(m, n, AB);
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nTime to init [A|B]: " << init_time << " s\n";

    // 5) Construct a sparse random S (8 nnz/col) in parallel
    int64_t nnz_per_col = 8;
    Kokkos::View<int*>    SrowIdx ("SrowIdx",  m*nnz_per_col);
    Kokkos::View<double*> Svals   ("Svals",    m*nnz_per_col);

    auto t2 = std::chrono::high_resolution_clock::now();
    init_sparse_sk_kokkos(m, sk_dim, nnz_per_col, SrowIdx, Svals);
    auto t3 = std::chrono::high_resolution_clock::now();
    double sampling_time = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "Time to sample sparse S: " << sampling_time << " s\n";

    // 6) Compute SAB = S * AB (sparse x dense)
    auto t4 = std::chrono::high_resolution_clock::now();
    spmm_sparse_kokkos(m, sk_dim, nnz_per_col, SrowIdx, Svals, AB, SAB);
    auto t5 = std::chrono::high_resolution_clock::now();
    double sketch_time = std::chrono::duration<double>(t5 - t4).count();
    std::cout << "Time to compute SAB = S*AB: " << sketch_time << " s\n";

    // 7) TLS on the sketched data
    auto t6 = std::chrono::high_resolution_clock::now();
    total_least_squares_cpu(sk_dim, n, SAB, sketch_x);
    auto t7 = std::chrono::high_resolution_clock::now();
    double sketched_tls_time = std::chrono::duration<double>(t7 - t6).count();
    std::cout << "Time for sketched TLS: " << sketched_tls_time << " s\n";

    double total_rand_time = sampling_time + sketch_time + sketched_tls_time;
    std::cout << "Total randomized TLS time: " << total_rand_time << " s\n\n";

    // 8) Classical TLS on full AB
    auto t8 = std::chrono::high_resolution_clock::now();
    total_least_squares_cpu(m, n, AB, true_x);
    auto t9 = std::chrono::high_resolution_clock::now();
    double classical_time = std::chrono::duration<double>(t9 - t8).count();
    std::cout << "Time for classical TLS: " << classical_time << " s\n";

    double speedup = (classical_time > 0.0 ? classical_time / total_rand_time : 0.0);
    std::cout << "Speedup of sketched vs. classical: " << speedup << "\n";

    // 9) Compare solutions
    //    delta = sketch_x - true_x
    Kokkos::View<double*> delta("delta", n);
    KokkosBlas::axpby(1.0, sketch_x, 0.0, delta);    // delta = sketch_x
    KokkosBlas::axpby(-1.0, true_x, 1.0, delta);     // delta -= true_x

    Kokkos::View<double> dev_nrm_delta("nrm_delta");
    Kokkos::View<double> dev_nrm_true ("nrm_true");
    KokkosBlas::nrm2(dev_nrm_delta, delta);
    KokkosBlas::nrm2(dev_nrm_true,  true_x);

    double h_delta = 0.0, h_true = 0.0;
    Kokkos::deep_copy(h_delta, dev_nrm_delta);
    Kokkos::deep_copy(h_true,  dev_nrm_true);

    std::cout << "||sketch_x - true_x|| / ||true_x|| = " 
              << (h_delta / h_true) << "\n\n";
  }
  Kokkos::finalize();
  return 0;
}
