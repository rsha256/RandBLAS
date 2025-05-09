.. :sd_hide_title:
.. ^ uncomment that if you want to prevent the header from rendering.

##########
Tutorial
##########

RandBLAS facilitates implementation of randomized numerical linear algebra (RandNLA) algorithms that need linear dimension-reduction maps.
These dimension-reduction maps, called *sketching operators*, are sampled at random from some prescribed distribution.
Once a sketching operator is sampled it is applied to a user provided *data matrix* to produce a smaller matrix called a *sketch*.

Abstractly, a sketch is supposed to summarize some geometric information that underlies its data matrix.
The RandNLA literature documents a huge array of possibilities for how to compute and process sketches to obtain various desired outcomes.
It also documents sketching operators of different "flavors;" some are sparse matrices, some are subsampled FFT-like operations, and others still are dense matrices. 

.. note::

  If we call something an "operator," we mean it's a *sketching operator* unless otherwise stated.

RandBLAS, at a glance
  It's useful to think of RandBLAS' sketching workflow in three steps.

    1. Get your hands on a random state.
    2. Define a sketching distribution, and use the random state to sample an operator from that distribution.
    3. Apply the operator with a function that's *almost* identical to `GEMM <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3>`_.

  To illustrate this workflow, suppose we have a 20,000-by-10,000 double-precision matrix :math:`A`  stored in column-major
  layout. Suppose also that we want to compute a sketch of the form :math:`B = AS`, where :math:`S` is a Gaussian matrix of size 10,000-by-50.
  This can be done as follows.

  .. code:: c++

    // step 1
    RandBLAS::RNGState state();
    // step 2
    RandBLAS::DenseDist D(10000, 50);
    RandBLAS::DenseSkOp<double> S(D, state);
    // step 3
    double B* = new double[20000 * 50];
    RandBLAS::sketch_general(
          blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          20000, 50,  10000,
          1.0, A, 20000, S, 0.0, B, 20000
    ); // B = AS

RandBLAS has a wealth of capabilities that are not reflected in that code snippet.
For example, it lets you set an integer-valued the seed when defining :math:`\texttt{state}`, and it provides a wide range of both dense and sparse sketching operators.
It even lets you compute products against *submatrices* of sketching operators without ever forming the full operator in memory.


.. toctree::
    :maxdepth: 4

    Background on GEMM <gemm>
    Defining a sketching distribution <distributions>
    Sampling a sketching operator <sampling_skops>
    Updating a sketch <sketch_updates>
    The meaning of "submat(・)" in RandBLAS documentation <submatrices>
    Memory management <memory>
