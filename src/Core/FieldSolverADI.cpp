#include "FieldSolverADI.h"
#include "Field.h"
#include "Beam.h"

#include <cmath>
#include <cstddef>

#ifdef GENESIS_USE_AMREX
#include "Genesis4BeamSoA.h"
#include "Genesis4FieldSoA.h"
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#endif

#ifndef GENESIS_FIELD_ADI_USE_SORT_REDUCE_SOURCE
#define GENESIS_FIELD_ADI_USE_SORT_REDUCE_SOURCE 1
#endif

#ifndef GENESIS_FIELD_ADI_USE_PCR
#define GENESIS_FIELD_ADI_USE_PCR 1
#endif

#ifndef GENESIS_FIELD_ADI_PCR_MAX_NGRID
#define GENESIS_FIELD_ADI_PCR_MAX_NGRID 512
#endif

#ifdef GENESIS_USE_AMREX
namespace {

struct G4ComplexPair {
  double re;
  double im;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_make_complex(double re, double im) noexcept
{
  return {re, im};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_add(G4ComplexPair a, G4ComplexPair b) noexcept
{
  return {a.re + b.re, a.im + b.im};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_sub(G4ComplexPair a, G4ComplexPair b) noexcept
{
  return {a.re - b.re, a.im - b.im};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_mul(G4ComplexPair a, G4ComplexPair b) noexcept
{
  return {a.re * b.re - a.im * b.im,
          a.re * b.im + a.im * b.re};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_div(G4ComplexPair a, G4ComplexPair b) noexcept
{
  const double den = b.re * b.re + b.im * b.im;
  return {(a.re * b.re + a.im * b.im) / den,
          (a.im * b.re - a.re * b.im) / den};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_neg(G4ComplexPair a) noexcept
{
  return {-a.re, -a.im};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_scale(G4ComplexPair a, double s) noexcept
{
  return {a.re * s, a.im * s};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
G4ComplexPair g4_mul_i_scale(G4ComplexPair a, double s) noexcept
{
  // (i*s) * (a.re + i*a.im) = -s*a.im + i*s*a.re
  return {-s * a.im, s * a.re};
}

void genesis4_fieldsolveradi_zero_sources_gpu(double* src_re,
                                               double* src_im,
                                               int total_cells)
{
  amrex::ParallelFor(total_cells,
    [=] AMREX_GPU_DEVICE (int i) noexcept {
      src_re[i] = 0.0;
      src_im[i] = 0.0;
    });
}

void genesis4_fieldsolveradi_build_source_gpu(int total_particles,
                                               int nslice,
                                               int ngrid,
                                               int first,
                                               int harm,
                                               double gridmax,
                                               double dgrid,
                                               double ax,
                                               double ay,
                                               double kx,
                                               double ky,
                                               double gradx,
                                               double grady,
                                               const double* slice_scl,
                                               const double* x,
                                               const double* y,
                                               const double* gamma,
                                               const double* theta,
                                               const int* slice_id,
                                               double* src_re,
                                               double* src_im)
{
  const int plane = ngrid * ngrid;

  amrex::ParallelFor(total_particles,
    [=] AMREX_GPU_DEVICE (int ip) noexcept {
      const int beam_slice = slice_id[ip];
      if (beam_slice < 0 || beam_slice >= nslice) { return; }

      const double scl = slice_scl[beam_slice];
      if (scl == 0.0) { return; }

      const double xp = x[ip];
      const double yp = y[ip];
      if (!(xp > -gridmax && xp < gridmax && yp > -gridmax && yp < gridmax)) {
        return;
      }

      double wx_raw = (xp + gridmax) / dgrid;
      double wy_raw = (yp + gridmax) / dgrid;
      const int ix = static_cast<int>(floor(wx_raw));
      const int iy = static_cast<int>(floor(wy_raw));

      if (ix < 0 || ix >= ngrid - 1 || iy < 0 || iy >= ngrid - 1) {
        return;
      }

      const double wx = 1.0 + floor(wx_raw) - wx_raw;
      const double wy = 1.0 + floor(wy_raw) - wy_raw;

      const double dx = xp - ax;
      const double dy = yp - ay;
      const double faw2 = 1.0 + kx * dx * dx + ky * dy * dy
                        + 2.0 * (gradx * dx + grady * dy);
      const double part = sqrt(faw2) * scl / gamma[ip];
      const double phase = static_cast<double>(harm) * theta[ip];
      const double cpart_re = sin(phase) * part;
      const double cpart_im = cos(phase) * part;

      const int target_slice = (beam_slice + first) % nslice;
      const int base = target_slice * plane + ix + iy * ngrid;

      const double w00 = wx * wy;
      const double w10 = (1.0 - wx) * wy;
      const double w01 = wx * (1.0 - wy);
      const double w11 = (1.0 - wx) * (1.0 - wy);

      amrex::Gpu::Atomic::Add(&src_re[base], cpart_re * w00);
      amrex::Gpu::Atomic::Add(&src_im[base], cpart_im * w00);

      amrex::Gpu::Atomic::Add(&src_re[base + 1], cpart_re * w10);
      amrex::Gpu::Atomic::Add(&src_im[base + 1], cpart_im * w10);

      amrex::Gpu::Atomic::Add(&src_re[base + ngrid], cpart_re * w01);
      amrex::Gpu::Atomic::Add(&src_im[base + ngrid], cpart_im * w01);

      amrex::Gpu::Atomic::Add(&src_re[base + ngrid + 1], cpart_re * w11);
      amrex::Gpu::Atomic::Add(&src_im[base + ngrid + 1], cpart_im * w11);
    });
}

struct G4ZipComplexAdd {
  template <typename Tuple>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  Tuple operator()(Tuple const& a, Tuple const& b) const noexcept
  {
    return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                              thrust::get<1>(a) + thrust::get<1>(b));
  }
};

void genesis4_fieldsolveradi_build_source_contribs_gpu(int total_particles,
                                                        int nslice,
                                                        int ngrid,
                                                        int first,
                                                        int harm,
                                                        double gridmax,
                                                        double dgrid,
                                                        double ax,
                                                        double ay,
                                                        double kx,
                                                        double ky,
                                                        double gradx,
                                                        double grady,
                                                        const double* slice_scl,
                                                        const double* x,
                                                        const double* y,
                                                        const double* gamma,
                                                        const double* theta,
                                                        const int* slice_id,
                                                        int* contrib_key,
                                                        double* contrib_re,
                                                        double* contrib_im)
{
  const int plane = ngrid * ngrid;

  amrex::ParallelFor(total_particles,
    [=] AMREX_GPU_DEVICE (int ip) noexcept {
      const int out0 = 4 * ip;
      contrib_key[out0 + 0] = -1;
      contrib_key[out0 + 1] = -1;
      contrib_key[out0 + 2] = -1;
      contrib_key[out0 + 3] = -1;
      contrib_re[out0 + 0] = 0.0;
      contrib_re[out0 + 1] = 0.0;
      contrib_re[out0 + 2] = 0.0;
      contrib_re[out0 + 3] = 0.0;
      contrib_im[out0 + 0] = 0.0;
      contrib_im[out0 + 1] = 0.0;
      contrib_im[out0 + 2] = 0.0;
      contrib_im[out0 + 3] = 0.0;

      const int beam_slice = slice_id[ip];
      if (beam_slice < 0 || beam_slice >= nslice) { return; }

      const double scl = slice_scl[beam_slice];
      if (scl == 0.0) { return; }

      const double xp = x[ip];
      const double yp = y[ip];
      if (!(xp > -gridmax && xp < gridmax && yp > -gridmax && yp < gridmax)) {
        return;
      }

      const double wx_raw = (xp + gridmax) / dgrid;
      const double wy_raw = (yp + gridmax) / dgrid;
      const int ix = static_cast<int>(floor(wx_raw));
      const int iy = static_cast<int>(floor(wy_raw));

      if (ix < 0 || ix >= ngrid - 1 || iy < 0 || iy >= ngrid - 1) {
        return;
      }

      const double wx = 1.0 + static_cast<double>(ix) - wx_raw;
      const double wy = 1.0 + static_cast<double>(iy) - wy_raw;

      const double dx = xp - ax;
      const double dy = yp - ay;
      const double faw2 = 1.0 + kx * dx * dx + ky * dy * dy
                        + 2.0 * (gradx * dx + grady * dy);
      const double part = sqrt(faw2) * scl / gamma[ip];
      const double phase = static_cast<double>(harm) * theta[ip];
      const double cpart_re = sin(phase) * part;
      const double cpart_im = cos(phase) * part;

      const int target_slice = (beam_slice + first) % nslice;
      const int base = target_slice * plane + ix + iy * ngrid;

      const double w00 = wx * wy;
      const double w10 = (1.0 - wx) * wy;
      const double w01 = wx * (1.0 - wy);
      const double w11 = (1.0 - wx) * (1.0 - wy);

      contrib_key[out0 + 0] = base;
      contrib_re [out0 + 0] = cpart_re * w00;
      contrib_im [out0 + 0] = cpart_im * w00;

      contrib_key[out0 + 1] = base + 1;
      contrib_re [out0 + 1] = cpart_re * w10;
      contrib_im [out0 + 1] = cpart_im * w10;

      contrib_key[out0 + 2] = base + ngrid;
      contrib_re [out0 + 2] = cpart_re * w01;
      contrib_im [out0 + 2] = cpart_im * w01;

      contrib_key[out0 + 3] = base + ngrid + 1;
      contrib_re [out0 + 3] = cpart_re * w11;
      contrib_im [out0 + 3] = cpart_im * w11;
    });
}

void genesis4_fieldsolveradi_add_reduced_source_gpu(int n_unique,
                                                     const int* reduced_key,
                                                     const double* reduced_re,
                                                     const double* reduced_im,
                                                     double* rhs_re,
                                                     double* rhs_im)
{
  amrex::ParallelFor(n_unique,
    [=] AMREX_GPU_DEVICE (int i) noexcept {
      const int key = reduced_key[i];
      if (key >= 0) {
        rhs_re[key] += reduced_re[i];
        rhs_im[key] += reduced_im[i];
      }
    });
}

int genesis4_fieldsolveradi_build_source_sort_reduce_gpu(int total_particles,
                                                           int nslice,
                                                           int ngrid,
                                                           int first,
                                                           int harm,
                                                           double gridmax,
                                                           double dgrid,
                                                           double ax,
                                                           double ay,
                                                           double kx,
                                                           double ky,
                                                           double gradx,
                                                           double grady,
                                                           const double* slice_scl,
                                                           const double* x,
                                                           const double* y,
                                                           const double* gamma,
                                                           const double* theta,
                                                           const int* slice_id,
                                                           amrex::Gpu::DeviceVector<int>& d_contrib_key,
                                                           amrex::Gpu::DeviceVector<int>& d_reduce_key,
                                                           amrex::Gpu::DeviceVector<double>& d_contrib_re,
                                                           amrex::Gpu::DeviceVector<double>& d_contrib_im,
                                                           amrex::Gpu::DeviceVector<double>& d_reduce_re,
                                                           amrex::Gpu::DeviceVector<double>& d_reduce_im,
                                                           int& capacity)
{
  const int ncontrib = 4 * total_particles;
  if (ncontrib <= 0) { return 0; }

  if (capacity < ncontrib) {
    d_contrib_key.resize(ncontrib);
    d_reduce_key.resize(ncontrib);
    d_contrib_re.resize(ncontrib);
    d_contrib_im.resize(ncontrib);
    d_reduce_re.resize(ncontrib);
    d_reduce_im.resize(ncontrib);
    capacity = ncontrib;
  }

  genesis4_fieldsolveradi_build_source_contribs_gpu(total_particles,
                                                     nslice,
                                                     ngrid,
                                                     first,
                                                     harm,
                                                     gridmax,
                                                     dgrid,
                                                     ax,
                                                     ay,
                                                     kx,
                                                     ky,
                                                     gradx,
                                                     grady,
                                                     slice_scl,
                                                     x,
                                                     y,
                                                     gamma,
                                                     theta,
                                                     slice_id,
                                                     d_contrib_key.data(),
                                                     d_contrib_re.data(),
                                                     d_contrib_im.data());

  auto policy = thrust::cuda::par.on(amrex::Gpu::gpuStream());
  auto key_first = thrust::device_pointer_cast(d_contrib_key.data());
  auto key_last = key_first + ncontrib;
  auto val_first = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::device_pointer_cast(d_contrib_re.data()),
                         thrust::device_pointer_cast(d_contrib_im.data())));

  thrust::sort_by_key(policy, key_first, key_last, val_first);

  auto out_key_first = thrust::device_pointer_cast(d_reduce_key.data());
  auto out_val_first = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::device_pointer_cast(d_reduce_re.data()),
                         thrust::device_pointer_cast(d_reduce_im.data())));

  auto ends = thrust::reduce_by_key(policy,
                                    key_first,
                                    key_last,
                                    val_first,
                                    out_key_first,
                                    out_val_first,
                                    thrust::equal_to<int>(),
                                    G4ZipComplexAdd());

  return static_cast<int>(ends.first - out_key_first);
}

void genesis4_fieldsolveradi_build_rhs_y_laplacian_gpu(int total_cells,
                                                        int ngrid,
                                                        double cstep_im,
                                                        const double* field_re,
                                                        const double* field_im,
                                                        double* r_re,
                                                        double* r_im)
{
  const int plane = ngrid * ngrid;

  amrex::ParallelFor(total_cells,
    [=] AMREX_GPU_DEVICE (int g) noexcept {
      const int local = g % plane;
      const int iy = local / ngrid;

      G4ComplexPair lap = g4_make_complex(0.0, 0.0);
      const G4ComplexPair center = g4_make_complex(field_re[g], field_im[g]);

      if (iy == 0) {
        lap = g4_sub(g4_make_complex(field_re[g + ngrid], field_im[g + ngrid]),
                     g4_scale(center, 2.0));
      } else if (iy == ngrid - 1) {
        lap = g4_sub(g4_make_complex(field_re[g - ngrid], field_im[g - ngrid]),
                     g4_scale(center, 2.0));
      } else {
        lap = g4_add(g4_make_complex(field_re[g + ngrid], field_im[g + ngrid]),
                     g4_make_complex(field_re[g - ngrid], field_im[g - ngrid]));
        lap = g4_sub(lap, g4_scale(center, 2.0));
      }

      const G4ComplexPair rhs = g4_add(center, g4_mul_i_scale(lap, cstep_im));
      r_re[g] = rhs.re;
      r_im[g] = rhs.im;
    });
}

void genesis4_fieldsolveradi_tridagx_gpu(int nslice,
                                          int ngrid,
                                          const double* r_re,
                                          const double* r_im,
                                          const double* c_re,
                                          const double* c_im,
                                          const double* cbet_re,
                                          const double* cbet_im,
                                          const double* cwet_re,
                                          const double* cwet_im,
                                          double* field_re,
                                          double* field_im)
{
  const int nlines = nslice * ngrid;
  const int plane = ngrid * ngrid;

  amrex::ParallelFor(nlines,
    [=] AMREX_GPU_DEVICE (int line) noexcept {
      const int s = line / ngrid;
      const int row = line - s * ngrid;
      const int base = s * plane + row * ngrid;

      G4ComplexPair u = g4_mul(g4_make_complex(r_re[base], r_im[base]),
                               g4_make_complex(cbet_re[0], cbet_im[0]));
      field_re[base] = u.re;
      field_im[base] = u.im;

      for (int k = 1; k < ngrid; ++k) {
        const int idx = base + k;
        const G4ComplexPair ck = g4_make_complex(c_re[k], c_im[k]);
        const G4ComplexPair prev = g4_make_complex(field_re[idx - 1], field_im[idx - 1]);
        const G4ComplexPair tmp = g4_sub(g4_make_complex(r_re[idx], r_im[idx]),
                                         g4_mul(ck, prev));
        u = g4_mul(tmp, g4_make_complex(cbet_re[k], cbet_im[k]));
        field_re[idx] = u.re;
        field_im[idx] = u.im;
      }

      for (int k = ngrid - 2; k >= 0; --k) {
        const int idx = base + k;
        const G4ComplexPair corr = g4_mul(g4_make_complex(cwet_re[k + 1], cwet_im[k + 1]),
                                          g4_make_complex(field_re[idx + 1], field_im[idx + 1]));
        u = g4_sub(g4_make_complex(field_re[idx], field_im[idx]), corr);
        field_re[idx] = u.re;
        field_im[idx] = u.im;
      }
    });
}

void genesis4_fieldsolveradi_build_rhs_x_laplacian_gpu(int total_cells,
                                                        int ngrid,
                                                        double cstep_im,
                                                        const double* field_re,
                                                        const double* field_im,
                                                        double* r_re,
                                                        double* r_im)
{
  const int plane = ngrid * ngrid;

  amrex::ParallelFor(total_cells,
    [=] AMREX_GPU_DEVICE (int g) noexcept {
      const int local = g % plane;
      const int ix = local % ngrid;

      G4ComplexPair lap = g4_make_complex(0.0, 0.0);
      const G4ComplexPair center = g4_make_complex(field_re[g], field_im[g]);

      if (ix == 0) {
        lap = g4_sub(g4_make_complex(field_re[g + 1], field_im[g + 1]),
                     g4_scale(center, 2.0));
      } else if (ix == ngrid - 1) {
        lap = g4_sub(g4_make_complex(field_re[g - 1], field_im[g - 1]),
                     g4_scale(center, 2.0));
      } else {
        lap = g4_add(g4_make_complex(field_re[g + 1], field_im[g + 1]),
                     g4_make_complex(field_re[g - 1], field_im[g - 1]));
        lap = g4_sub(lap, g4_scale(center, 2.0));
      }

      const G4ComplexPair rhs = g4_add(center, g4_mul_i_scale(lap, cstep_im));
      r_re[g] = rhs.re;
      r_im[g] = rhs.im;
    });
}

void genesis4_fieldsolveradi_tridagy_gpu(int nslice,
                                          int ngrid,
                                          const double* r_re,
                                          const double* r_im,
                                          const double* c_re,
                                          const double* c_im,
                                          const double* cbet_re,
                                          const double* cbet_im,
                                          const double* cwet_re,
                                          const double* cwet_im,
                                          double* field_re,
                                          double* field_im)
{
  const int nlines = nslice * ngrid;
  const int plane = ngrid * ngrid;

  amrex::ParallelFor(nlines,
    [=] AMREX_GPU_DEVICE (int line) noexcept {
      const int s = line / ngrid;
      const int col = line - s * ngrid;
      const int base = s * plane + col;

      G4ComplexPair u = g4_mul(g4_make_complex(r_re[base], r_im[base]),
                               g4_make_complex(cbet_re[0], cbet_im[0]));
      field_re[base] = u.re;
      field_im[base] = u.im;

      for (int k = 1; k < ngrid; ++k) {
        const int idx = base + k * ngrid;
        const G4ComplexPair ck = g4_make_complex(c_re[k], c_im[k]);
        const G4ComplexPair prev = g4_make_complex(field_re[idx - ngrid], field_im[idx - ngrid]);
        const G4ComplexPair tmp = g4_sub(g4_make_complex(r_re[idx], r_im[idx]),
                                         g4_mul(ck, prev));
        u = g4_mul(tmp, g4_make_complex(cbet_re[k], cbet_im[k]));
        field_re[idx] = u.re;
        field_im[idx] = u.im;
      }

      for (int k = ngrid - 2; k >= 0; --k) {
        const int idx = base + k * ngrid;
        const G4ComplexPair corr = g4_mul(g4_make_complex(cwet_re[k + 1], cwet_im[k + 1]),
                                          g4_make_complex(field_re[idx + ngrid], field_im[idx + ngrid]));
        u = g4_sub(g4_make_complex(field_re[idx], field_im[idx]), corr);
        field_re[idx] = u.re;
        field_im[idx] = u.im;
      }
    });
}

void genesis4_fieldsolveradi_prepare_pcr_factors(int ngrid,
                                                   double rtmp,
                                                   amrex::Gpu::ManagedVector<double>& alpha_re,
                                                   amrex::Gpu::ManagedVector<double>& alpha_im,
                                                   amrex::Gpu::ManagedVector<double>& beta_re,
                                                   amrex::Gpu::ManagedVector<double>& beta_im,
                                                   amrex::Gpu::ManagedVector<double>& b_re,
                                                   amrex::Gpu::ManagedVector<double>& b_im,
                                                   int& factor_ngrid,
                                                   int& num_stages)
{
  num_stages = 0;
  for (int stride = 1; stride < ngrid; stride <<= 1) {
    ++num_stages;
  }

  alpha_re.resize(num_stages * ngrid);
  alpha_im.resize(num_stages * ngrid);
  beta_re.resize(num_stages * ngrid);
  beta_im.resize(num_stages * ngrid);
  b_re.resize(ngrid);
  b_im.resize(ngrid);

  std::vector<G4ComplexPair> a(ngrid);
  std::vector<G4ComplexPair> b(ngrid);
  std::vector<G4ComplexPair> c(ngrid);
  std::vector<G4ComplexPair> na(ngrid);
  std::vector<G4ComplexPair> nb(ngrid);
  std::vector<G4ComplexPair> nc(ngrid);

  for (int i = 0; i < ngrid; ++i) {
    a[i] = (i == 0) ? g4_make_complex(0.0, 0.0) : g4_make_complex(0.0, -rtmp);
    b[i] = g4_make_complex(1.0, 2.0 * rtmp);
    c[i] = (i == ngrid - 1) ? g4_make_complex(0.0, 0.0) : g4_make_complex(0.0, -rtmp);
  }

  int stage = 0;
  for (int stride = 1; stride < ngrid; stride <<= 1, ++stage) {
    for (int tid = 0; tid < ngrid; ++tid) {
      G4ComplexPair alpha = g4_make_complex(0.0, 0.0);
      G4ComplexPair beta = g4_make_complex(0.0, 0.0);
      G4ComplexPair a_new = g4_make_complex(0.0, 0.0);
      G4ComplexPair b_new = b[tid];
      G4ComplexPair c_new = g4_make_complex(0.0, 0.0);

      if (tid >= stride) {
        alpha = g4_neg(g4_div(a[tid], b[tid - stride]));
        a_new = g4_mul(alpha, a[tid - stride]);
        b_new = g4_add(b_new, g4_mul(alpha, c[tid - stride]));
      }
      if (tid + stride < ngrid) {
        beta = g4_neg(g4_div(c[tid], b[tid + stride]));
        c_new = g4_mul(beta, c[tid + stride]);
        b_new = g4_add(b_new, g4_mul(beta, a[tid + stride]));
      }

      const int off = stage * ngrid + tid;
      alpha_re[off] = alpha.re;
      alpha_im[off] = alpha.im;
      beta_re[off] = beta.re;
      beta_im[off] = beta.im;
      na[tid] = a_new;
      nb[tid] = b_new;
      nc[tid] = c_new;
    }

    a.swap(na);
    b.swap(nb);
    c.swap(nc);
  }

  for (int i = 0; i < ngrid; ++i) {
    b_re[i] = b[i].re;
    b_im[i] = b[i].im;
  }

  factor_ngrid = ngrid;
}

__global__ void genesis4_fieldsolveradi_tridagx_pcr_kernel(int nslice,
                                                        int ngrid,
                                                        int num_stages,
                                                        const double* __restrict__ alpha_re,
                                                        const double* __restrict__ alpha_im,
                                                        const double* __restrict__ beta_re,
                                                        const double* __restrict__ beta_im,
                                                        const double* __restrict__ b_re,
                                                        const double* __restrict__ b_im,
                                                        const double* __restrict__ r_re,
                                                        const double* __restrict__ r_im,
                                                        double* __restrict__ field_re,
                                                        double* __restrict__ field_im)
{
  extern __shared__ G4ComplexPair shared[];
  G4ComplexPair* cur = shared;
  G4ComplexPair* nxt = cur + blockDim.x;

  const int tid = threadIdx.x;
  const int line = blockIdx.x;
  const int plane = ngrid * ngrid;
  const int s = line / ngrid;
  const int row = line - s * ngrid;
  const int base = s * plane + row * ngrid;

  if (tid < ngrid) {
    const int idx = base + tid;
    cur[tid] = g4_make_complex(r_re[idx], r_im[idx]);
  }
  __syncthreads();

  int stride = 1;
  for (int stage = 0; stage < num_stages; ++stage, stride <<= 1) {
    if (tid < ngrid) {
      const int off = stage * ngrid + tid;
      G4ComplexPair d = cur[tid];
      if (tid >= stride) {
        d = g4_add(d, g4_mul(g4_make_complex(alpha_re[off], alpha_im[off]),
                             cur[tid - stride]));
      }
      if (tid + stride < ngrid) {
        d = g4_add(d, g4_mul(g4_make_complex(beta_re[off], beta_im[off]),
                             cur[tid + stride]));
      }
      nxt[tid] = d;
    }
    __syncthreads();
    G4ComplexPair* tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  if (tid < ngrid) {
    const int idx = base + tid;
    const G4ComplexPair u = g4_div(cur[tid], g4_make_complex(b_re[tid], b_im[tid]));
    field_re[idx] = u.re;
    field_im[idx] = u.im;
  }
}

__global__ void genesis4_fieldsolveradi_tridagy_pcr_kernel(int nslice,
                                                        int ngrid,
                                                        int num_stages,
                                                        const double* __restrict__ alpha_re,
                                                        const double* __restrict__ alpha_im,
                                                        const double* __restrict__ beta_re,
                                                        const double* __restrict__ beta_im,
                                                        const double* __restrict__ b_re,
                                                        const double* __restrict__ b_im,
                                                        const double* __restrict__ r_re,
                                                        const double* __restrict__ r_im,
                                                        double* __restrict__ field_re,
                                                        double* __restrict__ field_im)
{
  extern __shared__ G4ComplexPair shared[];
  G4ComplexPair* cur = shared;
  G4ComplexPair* nxt = cur + blockDim.x;

  const int tid = threadIdx.x;
  const int line = blockIdx.x;
  const int plane = ngrid * ngrid;
  const int s = line / ngrid;
  const int col = line - s * ngrid;
  const int base = s * plane + col;

  if (tid < ngrid) {
    const int idx = base + tid * ngrid;
    cur[tid] = g4_make_complex(r_re[idx], r_im[idx]);
  }
  __syncthreads();

  int stride = 1;
  for (int stage = 0; stage < num_stages; ++stage, stride <<= 1) {
    if (tid < ngrid) {
      const int off = stage * ngrid + tid;
      G4ComplexPair d = cur[tid];
      if (tid >= stride) {
        d = g4_add(d, g4_mul(g4_make_complex(alpha_re[off], alpha_im[off]),
                             cur[tid - stride]));
      }
      if (tid + stride < ngrid) {
        d = g4_add(d, g4_mul(g4_make_complex(beta_re[off], beta_im[off]),
                             cur[tid + stride]));
      }
      nxt[tid] = d;
    }
    __syncthreads();
    G4ComplexPair* tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  if (tid < ngrid) {
    const int idx = base + tid * ngrid;
    const G4ComplexPair u = g4_div(cur[tid], g4_make_complex(b_re[tid], b_im[tid]));
    field_re[idx] = u.re;
    field_im[idx] = u.im;
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int g4_next_power_of_two_host_device(int n) noexcept
{
  int p = 1;
  while (p < n) { p <<= 1; }
  return p;
}

bool genesis4_fieldsolveradi_tridagx_pcr_gpu(int nslice,
                                               int ngrid,
                                               int num_stages,
                                               const double* alpha_re,
                                               const double* alpha_im,
                                               const double* beta_re,
                                               const double* beta_im,
                                               const double* b_re,
                                               const double* b_im,
                                               const double* r_re,
                                               const double* r_im,
                                               double* field_re,
                                               double* field_im)
{
  if (ngrid <= 1 || ngrid > GENESIS_FIELD_ADI_PCR_MAX_NGRID || num_stages <= 0) {
    return false;
  }
  const int block = g4_next_power_of_two_host_device(ngrid);
  const int nlines = nslice * ngrid;
  const std::size_t shmem = 2 * static_cast<std::size_t>(block) * sizeof(G4ComplexPair);
  genesis4_fieldsolveradi_tridagx_pcr_kernel<<<nlines, block, shmem, amrex::Gpu::gpuStream()>>>(
      nslice, ngrid, num_stages,
      alpha_re, alpha_im, beta_re, beta_im, b_re, b_im,
      r_re, r_im, field_re, field_im);
  return cudaGetLastError() == cudaSuccess;
}

bool genesis4_fieldsolveradi_tridagy_pcr_gpu(int nslice,
                                               int ngrid,
                                               int num_stages,
                                               const double* alpha_re,
                                               const double* alpha_im,
                                               const double* beta_re,
                                               const double* beta_im,
                                               const double* b_re,
                                               const double* b_im,
                                               const double* r_re,
                                               const double* r_im,
                                               double* field_re,
                                               double* field_im)
{
  if (ngrid <= 1 || ngrid > GENESIS_FIELD_ADI_PCR_MAX_NGRID || num_stages <= 0) {
    return false;
  }
  const int block = g4_next_power_of_two_host_device(ngrid);
  const int nlines = nslice * ngrid;
  const std::size_t shmem = 2 * static_cast<std::size_t>(block) * sizeof(G4ComplexPair);
  genesis4_fieldsolveradi_tridagy_pcr_kernel<<<nlines, block, shmem, amrex::Gpu::gpuStream()>>>(
      nslice, ngrid, num_stages,
      alpha_re, alpha_im, beta_re, beta_im, b_re, b_im,
      r_re, r_im, field_re, field_im);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace
#endif

FieldSolverADI::~FieldSolverADI() = default;

void FieldSolverADI::advance(double delz, Field *field, Beam *beam, Undulator *und) {
#ifdef GENESIS_USE_AMREX
  this->advance_gpu(delz, field, beam, und);
#else
  this->advance_cpu(delz, field, beam, und);
#endif
}

void FieldSolverADI::advance_cpu(double delz, Field *field, Beam *beam, Undulator *und) {

    for (unsigned long ii = 0; ii < field->field.size(); ii++) {  // ii is index for the beam

    // clear source term
        for (int ig = 0; ig < ngrid * ngrid; ig++) {
      crsource[ig] = 0;
    }

    // constructing source term
    int harm = field->getHarm();
    if (und->inUndulator() && field->isEnabled() && (harm % 2 == 1)) { // do not need to calculate for even harmonics
      double scl = und->fc(harm) * vacimp * beam->current[ii] * field->xks * delz;
      scl /= 4 * eev * static_cast<double>(beam->beam[ii].size()) * field->dgrid * field->dgrid;
      complex<double> cpart;
      double part, weight, wx, wy;
      int idx;

      for (auto & particle : beam->beam.at(ii)) {
        double x = particle.x;
        double y = particle.y;
        double theta = static_cast<double>(harm) * particle.theta;
        double gamma = particle.gamma;

        if (field->getLLGridpoint(x, y, &wx, &wy, &idx)) {

          part = sqrt(und->faw2(x, y)) * scl / gamma;
                    // tmp  should be also normalized with beta parallel
          cpart = complex<double>(sin(theta), cos(theta)) * part;

          weight = wx * wy;
          crsource[idx] += weight * cpart;
          weight = (1 - wx) * wy;
          idx++;
          crsource[idx] += weight * cpart;
          weight = wx * (1 - wy);
          idx += ngrid - 1;
          crsource[idx] += weight * cpart;
          weight = (1 - wx) * (1 - wy);
          idx++;
          crsource[idx] += weight * cpart;
        }
      }
        }  // end of source term construction

        unsigned long i = (ii + field->first) % field->field.size();           // index for the field
    this->ADI(field->field[i]);
  }
}

#ifdef GENESIS_USE_AMREX
void FieldSolverADI::advance_gpu(double delz, Field *field, Beam *beam, Undulator *und) {

  if (field == nullptr || beam == nullptr || und == nullptr) {
    amrex::Abort("FieldSolverADI::advance_gpu received a null field/beam/undulator pointer");
  }

  Genesis4BeamSoA* bsoa = beam->beamSoA;
  Genesis4FieldSoA* fsoa = field->fieldSoA;

  if (bsoa == nullptr || !bsoa->initialized ||
      fsoa == nullptr || !fsoa->initialized ||
      ngrid == 0 || field->ngrid != static_cast<int>(ngrid)) {
    amrex::Abort("FieldSolverADI::advance_gpu requires initialized BeamSoA/FieldSoA and matching ngrid");
  }

  const int nslice = fsoa->nslice;
  const int ngrid_i = static_cast<int>(ngrid);
  const int plane = ngrid_i * ngrid_i;
  const int total_cells = nslice * plane;
  const int total_particles = bsoa->total_particles;

  if (nslice <= 0 || plane <= 0 || total_cells <= 0 ||
      bsoa->nslice != nslice ||
      fsoa->ngrid != ngrid_i ||
      static_cast<int>(fsoa->field_re.size()) < total_cells ||
      static_cast<int>(fsoa->field_im.size()) < total_cells ||
      static_cast<int>(bsoa->x.size()) < total_particles ||
      static_cast<int>(bsoa->y.size()) < total_particles ||
      static_cast<int>(bsoa->gamma.size()) < total_particles ||
      static_cast<int>(bsoa->theta.size()) < total_particles ||
      static_cast<int>(bsoa->slice_id.size()) < total_particles ||
      static_cast<int>(bsoa->slice_offsets.size()) < nslice + 1 ||
      static_cast<int>(beam->current.size()) < nslice) {
    amrex::Abort("FieldSolverADI::advance_gpu found inconsistent SoA dimensions");
  }

  if (d_alloc_nslice != nslice || d_alloc_cells_per_slice != plane) {
    d_r_re.resize(total_cells);
    d_r_im.resize(total_cells);
    d_slice_scl.resize(nslice);
    d_alloc_nslice = nslice;
    d_alloc_cells_per_slice = plane;
  }

  const bool pcr_factor_required = (ngrid_i > 1 && ngrid_i <= GENESIS_FIELD_ADI_PCR_MAX_NGRID);
  if (static_cast<int>(d_c_re.size()) < ngrid_i ||
      static_cast<int>(d_cbet_re.size()) < ngrid_i ||
      static_cast<int>(d_cwet_re.size()) < ngrid_i ||
      d_pcr_factor_ngrid != ngrid_i ||
      static_cast<int>(d_pcr_b_re.size()) < ngrid_i ||
      (pcr_factor_required &&
       (d_pcr_num_stages <= 0 ||
        static_cast<int>(d_pcr_alpha_re.size()) < d_pcr_num_stages * ngrid_i ||
        static_cast<int>(d_pcr_beta_re.size()) < d_pcr_num_stages * ngrid_i))) {
    amrex::Abort("FieldSolverADI::advance_gpu was called before GPU ADI coefficients/PCR factors were initialized");
  }

  amrex::Gpu::HostVector<int> h_slice_offsets(nslice + 1);
  amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                        bsoa->slice_offsets.begin(),
                        bsoa->slice_offsets.begin() + nslice + 1,
                        h_slice_offsets.begin());
  amrex::Gpu::streamSynchronize();

  const int harm = field->getHarm();
  const bool do_source = und->inUndulator() && field->isEnabled() && ((harm % 2) == 1);

  for (int is = 0; is < nslice; ++is) {
    double scl = 0.0;
    if (do_source) {
      const int npar = h_slice_offsets[is + 1] - h_slice_offsets[is];
      if (npar > 0) {
        scl = und->fc(harm) * vacimp * beam->current[is] * field->xks * delz;
        scl /= 4.0 * eev * static_cast<double>(npar) * field->dgrid * field->dgrid;
      }
    }
    d_slice_scl[is] = scl;
  }

  double ax = 0.0;
  double ay = 0.0;
  double kx = 0.0;
  double ky = 0.0;
  double gradx = 0.0;
  double grady = 0.0;
  const int istep = und->getStep();
  if (do_source && istep >= 0 && istep < static_cast<int>(und->ax.size())) {
    ax = und->ax[istep];
    ay = und->ay[istep];
    kx = und->kx[istep];
    ky = und->ky[istep];
    gradx = und->gradx[istep];
    grady = und->grady[istep];
  }

  double* r_re = d_r_re.data();
  double* r_im = d_r_im.data();
  double* fld_re = fsoa->field_re.data();
  double* fld_im = fsoa->field_im.data();

#if GENESIS_FIELD_ADI_USE_SORT_REDUCE_SOURCE
  int n_unique_source = 0;
  if (do_source && total_particles > 0) {
    n_unique_source = genesis4_fieldsolveradi_build_source_sort_reduce_gpu(total_particles,
                                                                            nslice,
                                                                            ngrid_i,
                                                                            field->first,
                                                                            harm,
                                                                            field->gridmax,
                                                                            field->dgrid,
                                                                            ax,
                                                                            ay,
                                                                            kx,
                                                                            ky,
                                                                            gradx,
                                                                            grady,
                                                                            d_slice_scl.data(),
                                                                            bsoa->x.data(),
                                                                            bsoa->y.data(),
                                                                            bsoa->gamma.data(),
                                                                            bsoa->theta.data(),
                                                                            bsoa->slice_id.data(),
                                                                            d_contrib_key,
                                                                            d_reduce_key,
                                                                            d_contrib_re,
                                                                            d_contrib_im,
                                                                            d_reduce_re,
                                                                            d_reduce_im,
                                                                            d_source_contrib_capacity);
  }
#endif

  genesis4_fieldsolveradi_build_rhs_y_laplacian_gpu(total_cells, ngrid_i, cstep.imag(),
                                                    fld_re, fld_im,
                                                    r_re, r_im);
#if GENESIS_FIELD_ADI_USE_SORT_REDUCE_SOURCE
  if (n_unique_source > 0) {
    genesis4_fieldsolveradi_add_reduced_source_gpu(n_unique_source,
                                                    d_reduce_key.data(),
                                                    d_reduce_re.data(),
                                                    d_reduce_im.data(),
                                                    r_re,
                                                    r_im);
  }
#else
  if (do_source && total_particles > 0) {
    genesis4_fieldsolveradi_build_source_gpu(total_particles,
                                             nslice,
                                             ngrid_i,
                                             field->first,
                                             harm,
                                             field->gridmax,
                                             field->dgrid,
                                             ax,
                                             ay,
                                             kx,
                                             ky,
                                             gradx,
                                             grady,
                                             d_slice_scl.data(),
                                             bsoa->x.data(),
                                             bsoa->y.data(),
                                             bsoa->gamma.data(),
                                             bsoa->theta.data(),
                                             bsoa->slice_id.data(),
                                             r_re,
                                             r_im);
  }
#endif
#if GENESIS_FIELD_ADI_USE_PCR
  if (!genesis4_fieldsolveradi_tridagx_pcr_gpu(nslice, ngrid_i, d_pcr_num_stages,
                                               d_pcr_alpha_re.data(), d_pcr_alpha_im.data(),
                                               d_pcr_beta_re.data(), d_pcr_beta_im.data(),
                                               d_pcr_b_re.data(), d_pcr_b_im.data(),
                                               r_re, r_im,
                                               fld_re, fld_im)) {
    genesis4_fieldsolveradi_tridagx_gpu(nslice, ngrid_i,
                                        r_re, r_im,
                                        d_c_re.data(), d_c_im.data(),
                                        d_cbet_re.data(), d_cbet_im.data(),
                                        d_cwet_re.data(), d_cwet_im.data(),
                                        fld_re, fld_im);
  }
#else
  genesis4_fieldsolveradi_tridagx_gpu(nslice, ngrid_i,
                                      r_re, r_im,
                                      d_c_re.data(), d_c_im.data(),
                                      d_cbet_re.data(), d_cbet_im.data(),
                                      d_cwet_re.data(), d_cwet_im.data(),
                                      fld_re, fld_im);
#endif

  genesis4_fieldsolveradi_build_rhs_x_laplacian_gpu(total_cells, ngrid_i, cstep.imag(),
                                                    fld_re, fld_im,
                                                    r_re, r_im);
#if GENESIS_FIELD_ADI_USE_SORT_REDUCE_SOURCE
  if (n_unique_source > 0) {
    genesis4_fieldsolveradi_add_reduced_source_gpu(n_unique_source,
                                                    d_reduce_key.data(),
                                                    d_reduce_re.data(),
                                                    d_reduce_im.data(),
                                                    r_re,
                                                    r_im);
  }
#else
  if (do_source && total_particles > 0) {
    genesis4_fieldsolveradi_build_source_gpu(total_particles,
                                             nslice,
                                             ngrid_i,
                                             field->first,
                                             harm,
                                             field->gridmax,
                                             field->dgrid,
                                             ax,
                                             ay,
                                             kx,
                                             ky,
                                             gradx,
                                             grady,
                                             d_slice_scl.data(),
                                             bsoa->x.data(),
                                             bsoa->y.data(),
                                             bsoa->gamma.data(),
                                             bsoa->theta.data(),
                                             bsoa->slice_id.data(),
                                             r_re,
                                             r_im);
  }
#endif
#if GENESIS_FIELD_ADI_USE_PCR
  if (!genesis4_fieldsolveradi_tridagy_pcr_gpu(nslice, ngrid_i, d_pcr_num_stages,
                                               d_pcr_alpha_re.data(), d_pcr_alpha_im.data(),
                                               d_pcr_beta_re.data(), d_pcr_beta_im.data(),
                                               d_pcr_b_re.data(), d_pcr_b_im.data(),
                                               r_re, r_im,
                                               fld_re, fld_im)) {
    genesis4_fieldsolveradi_tridagy_gpu(nslice, ngrid_i,
                                        r_re, r_im,
                                        d_c_re.data(), d_c_im.data(),
                                        d_cbet_re.data(), d_cbet_im.data(),
                                        d_cwet_re.data(), d_cwet_im.data(),
                                        fld_re, fld_im);
  }
#else
  genesis4_fieldsolveradi_tridagy_gpu(nslice, ngrid_i,
                                      r_re, r_im,
                                      d_c_re.data(), d_c_im.data(),
                                      d_cbet_re.data(), d_cbet_im.data(),
                                      d_cwet_re.data(), d_cwet_im.data(),
                                      fld_re, fld_im);
#endif

}
#endif

void FieldSolverADI::ADI(vector<complex<double> > &crfield)
{
  int ix,idx;
  // implicit direction in x
  for (idx=0;idx<ngrid;idx++){
    r[idx]=crsource[idx]+crfield[idx]+cstep*(crfield[idx+ngrid]-2.0*crfield[idx]);
  }
  for (idx=ngrid;idx<ngrid*(ngrid-1);idx++){
    r[idx]=crsource[idx]+crfield[idx]+cstep*(crfield[idx+ngrid]-2.0*crfield[idx]+crfield[idx-ngrid]);
  }
  for (idx=ngrid*(ngrid-1);idx<ngrid*ngrid;idx++){
    r[idx]=crsource[idx]+crfield[idx]+cstep*(crfield[idx-ngrid]-2.0*crfield[idx]);
  }

  // solve tridiagonal system in x
  this->tridagx(crfield);

  // implicit direction in y
  for(ix=0;ix<ngrid*ngrid;ix+=ngrid){
    idx=ix;
    r[idx]=crsource[idx]+crfield[idx]+cstep*(crfield[idx+1]-2.0*crfield[idx]);
    for(idx=ix+1;idx<ix+ngrid-1;idx++){
      r[idx]=crsource[idx]+crfield[idx]+cstep*(crfield[idx+1]-2.0*crfield[idx]+crfield[idx-1]);
    }
    idx=ix+ngrid-1;
    r[idx]=crsource[idx]+crfield[idx]+cstep*(crfield[idx-1]-2.0*crfield[idx]);
  }

  // solve tridiagonal system in y
  this->tridagy(crfield);

}


void FieldSolverADI::tridagx(vector<complex<double > > &u) {
    for (int i = 0; i < ngrid * ngrid; i += ngrid) {
    u[i] = r[i] * cbet[0];
        for (int k = 1; k < ngrid; k++) {
      u[k + i] = (r[k + i] - c[k] * u[k + i - 1]) * cbet[k];
    }
    for (int k = ngrid - 2; k >= 0; k--) {
      u[k + i] -= cwet[k + 1] * u[k + i + 1];
    }
  }
}

void FieldSolverADI::tridagy(vector<complex<double > > &u) {
    for (int i = 0; i < ngrid; i++) {
    u[i] = r[i] * cbet[0];
  }
    for (int k = 1; k < ngrid; k++) {
    int n = k * ngrid;
        for (int i = 0; i < ngrid; i++) {
      u[n + i] = (r[n + i] - c[k] * u[n + i - ngrid]) * cbet[k];
    }
  }
  for (int k = ngrid - 2; k >= 0; k--) {
    int n = k * ngrid;
        for (int i = 0; i < ngrid; i++) {
      u[n + i] -= cwet[k + 1] * u[n + i + ngrid];
    }
  }
}


void FieldSolverADI::init(double delz,double dgrid, double xks, unsigned int ngrid_in) {

    if (delz == delz_save) {
    return;
  }
  delz_save = delz;
  ngrid = ngrid_in;


  double rtmp = 0.25 * delz / (xks * dgrid * dgrid); //factor dz/(4 ks dx^2)
  cstep = complex<double>(0, rtmp);

  auto *mupp = new double[ngrid];
  auto *mmid = new double[ngrid];
  auto *mlow = new double[ngrid];
  auto *cwrk1 = new complex<double>[ngrid];
  auto *cwrk2 = new complex<double>[ngrid];
  if (c.size() != ngrid) {
    c.resize(ngrid);
    r.resize(ngrid * ngrid);
    cbet.resize(ngrid);
    cwet.resize(ngrid);
    crsource.resize(ngrid * ngrid);
  }

  mupp[0] = rtmp;
  mmid[0] = -2 * rtmp;
  mlow[0] = 0;
    for (int i = 1; i < (ngrid - 1); i++) {
    mupp[i] = rtmp;
    mmid[i] = -2 * rtmp;
    mlow[i] = rtmp;
  }
  mupp[ngrid - 1] = 0;
  mmid[ngrid - 1] = -2 * rtmp;
  mlow[ngrid - 1] = rtmp;

    for (int i = 0; i < ngrid; i++) {
    cwrk1[i] = complex<double>(0, -mupp[i]);
    cwrk2[i] = complex<double>(1, -mmid[i]);
    c[i] = complex<double>(0, -mlow[i]);
  }


  cbet[0] = 1. / cwrk2[0];
  cwet[0] = 0.;
    for (int i = 1; i < ngrid; i++) {
    cwet[i] = cwrk1[i - 1] * cbet[i - 1];
    cbet[i] = 1. / (cwrk2[i] - c[i] * cwet[i]);
  }

#ifdef GENESIS_USE_AMREX
  d_c_re.resize(ngrid);
  d_c_im.resize(ngrid);
  d_cbet_re.resize(ngrid);
  d_cbet_im.resize(ngrid);
  d_cwet_re.resize(ngrid);
  d_cwet_im.resize(ngrid);

  for (int i = 0; i < static_cast<int>(ngrid); ++i) {
    d_c_re[i] = c[i].real();
    d_c_im[i] = c[i].imag();
    d_cbet_re[i] = cbet[i].real();
    d_cbet_im[i] = cbet[i].imag();
    d_cwet_re[i] = cwet[i].real();
    d_cwet_im[i] = cwet[i].imag();
  }

  genesis4_fieldsolveradi_prepare_pcr_factors(static_cast<int>(ngrid),
                                              rtmp,
                                              d_pcr_alpha_re,
                                              d_pcr_alpha_im,
                                              d_pcr_beta_re,
                                              d_pcr_beta_im,
                                              d_pcr_b_re,
                                              d_pcr_b_im,
                                              d_pcr_factor_ngrid,
                                              d_pcr_num_stages);
#endif

  delete[] mupp;
  delete[] mmid;
  delete[] mlow;
  delete[] cwrk1;
  delete[] cwrk2;
}
