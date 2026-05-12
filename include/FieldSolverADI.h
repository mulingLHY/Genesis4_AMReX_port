#ifndef __GENESIS_FIELDSOLVERADI__
#define __GENESIS_FIELDSOLVERADI__

#include <vector>
#include <iostream>
#include <string>
#include <complex>

#ifdef GENESIS_USE_AMREX
#include <AMReX_GpuContainers.H>
#endif

class Field;
class Beam;

#include "Particle.h"
#include "Undulator.h"
#include "FieldSolver.h"

using namespace std;

class FieldSolverADI : public FieldSolver{
 public:
  ~FieldSolverADI();

  void init(double,double,double,unsigned int) override;
  void advance(double, Field *, Beam *, Undulator *) override;
  void initSourceFilter(double,double,double,bool) override;

 private:
  unsigned int ngrid {0};
  double delz_save {0};
  complex<double> cstep;

  // CPU work arrays kept for non-AMReX builds.
  vector< complex< double > > r,c,cbet,cwet,crsource;

  void advance_cpu(double, Field *, Beam *, Undulator *);
  void ADI(vector<complex< double > > &);
  void tridagx(vector<complex< double > > &);
  void tridagy(vector<complex< double > > &);

#ifdef GENESIS_USE_AMREX
  void advance_gpu(double, Field *, Beam *, Undulator *);

  // GPU coefficient/work arrays.  Complex numbers are stored as split real/imag
  // arrays so device kernels do not depend on std::complex device support.
  amrex::Gpu::ManagedVector<double> d_c_re, d_c_im;
  amrex::Gpu::ManagedVector<double> d_cbet_re, d_cbet_im;
  amrex::Gpu::ManagedVector<double> d_cwet_re, d_cwet_im;

  // Precomputed PCR elimination factors for the constant ADI tridiagonal
  // matrix.  Size = d_pcr_num_stages * ngrid for alpha/beta and ngrid for
  // final diagonal.  These keep each field line from recomputing identical
  // complex divisions in every GPU block.
  amrex::Gpu::ManagedVector<double> d_pcr_alpha_re, d_pcr_alpha_im;
  amrex::Gpu::ManagedVector<double> d_pcr_beta_re, d_pcr_beta_im;
  amrex::Gpu::ManagedVector<double> d_pcr_b_re, d_pcr_b_im;

  // Per-slice ADI RHS arrays.  Size = nslice * ngrid * ngrid.
  // These are device-only buffers; source terms are scattered into them before
  // each ADI half-step to avoid holding a second full-grid source array.
  amrex::Gpu::DeviceVector<double> d_r_re, d_r_im;

  // Source scale per beam slice.
  amrex::Gpu::ManagedVector<double> d_slice_scl;

  // Sort/reduce source construction scratch.  This is solver-owned to avoid
  // cross-field aliasing from function-local static device buffers.
  amrex::Gpu::DeviceVector<int> d_contrib_key;
  amrex::Gpu::DeviceVector<int> d_reduce_key;
  amrex::Gpu::DeviceVector<double> d_contrib_re;
  amrex::Gpu::DeviceVector<double> d_contrib_im;
  amrex::Gpu::DeviceVector<double> d_reduce_re;
  amrex::Gpu::DeviceVector<double> d_reduce_im;

  int d_alloc_nslice {0};
  int d_alloc_cells_per_slice {0};
  int d_source_contrib_capacity {0};
  int d_pcr_factor_ngrid {0};
  int d_pcr_num_stages {0};
#endif
};

inline void FieldSolverADI::initSourceFilter(double x,double y,double z ,bool t) { return;}

#endif
