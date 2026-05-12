//
// Created by reiche on 06.01.22.
// AMReX/SoA diagnostics port.
//
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <limits>
#include <sstream>
#include <vector>

#include <mpi.h>

#include "Diagnostic.h"
#include "Output.h"
#include "Setup.h"
#include "Beam.h"
#include "Field.h"
#include "Undulator.h"

#ifdef GENESIS_USE_AMREX
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#ifdef AMREX_USE_CUDA
#include <cuda_runtime.h>
#endif
#include "Genesis4BeamSoA.h"
#include "Genesis4FieldSoA.h"
#endif

using namespace std;

extern bool MPISingle;
extern const double vacimp;
extern const double eev;


#ifdef GENESIS_USE_AMREX
struct DiagnosticAMReXScratch {
  amrex::Gpu::DeviceVector<double> beam_stats_dev;
  amrex::Gpu::DeviceVector<double> beam_bre_dev;
  amrex::Gpu::DeviceVector<double> beam_bim_dev;
  amrex::Gpu::DeviceVector<double> field_stats_dev;
  amrex::Gpu::DeviceVector<double> field_center_re_dev;
  amrex::Gpu::DeviceVector<double> field_center_im_dev;

  std::vector<double> beam_stats_host;
  std::vector<double> beam_bre_host;
  std::vector<double> beam_bim_host;
  std::vector<double> field_stats_host;
  std::vector<double> field_center_re_host;
  std::vector<double> field_center_im_host;
};
#endif

namespace {

inline void g4_store_value(map<string, vector<double> > &container,
                           const string &key,
                           unsigned long index,
                           double value)
{
  auto it = container.find(key);
  if (it != container.end() && index < it->second.size()) {
    it->second[index] = value;
  }
}

inline double *g4_value_ptr(map<string, vector<double> > &container,
                            const string &key)
{
  auto it = container.find(key);
  if (it == container.end() || it->second.empty()) { return nullptr; }
  return it->second.data();
}

inline void g4_store_ptr(double *ptr, unsigned long index, double value)
{
  if (ptr != nullptr) { ptr[index] = value; }
}

inline double g4_safe_sqrt_var(double v)
{
  return sqrt(fabs(v));
}


static inline void g4_allreduce_sum_inplace(double *values, int count)
{
  int mpi_size = 1;
  if (!MPISingle) { MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); }
  if (mpi_size <= 1) { return; }

  double recv[16];
  if (count > static_cast<int>(sizeof(recv) / sizeof(recv[0]))) {
    std::vector<double> tmp(static_cast<std::size_t>(count));
    MPI_Allreduce(values, tmp.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < count; ++i) { values[i] = tmp[static_cast<std::size_t>(i)]; }
    return;
  }

  MPI_Allreduce(values, recv, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (int i = 0; i < count; ++i) { values[i] = recv[i]; }
}

#ifdef GENESIS_USE_AMREX
constexpr int G4_DIAG_THREADS = 256;

constexpr int G4_BEAM_STATS = 23;
constexpr int G4_B_G1     = 0;
constexpr int G4_B_G2     = 1;
constexpr int G4_B_X1     = 2;
constexpr int G4_B_X2     = 3;
constexpr int G4_B_Y1     = 4;
constexpr int G4_B_Y2     = 5;
constexpr int G4_B_PX1    = 6;
constexpr int G4_B_PX2    = 7;
constexpr int G4_B_PY1    = 8;
constexpr int G4_B_PY2    = 9;
constexpr int G4_B_XPX    = 10;
constexpr int G4_B_YPY    = 11;
constexpr int G4_B_COUNT  = 12;
constexpr int G4_B_XMIN   = 13;
constexpr int G4_B_XMAX   = 14;
constexpr int G4_B_PXMIN  = 15;
constexpr int G4_B_PXMAX  = 16;
constexpr int G4_B_YMIN   = 17;
constexpr int G4_B_YMAX   = 18;
constexpr int G4_B_PYMIN  = 19;
constexpr int G4_B_PYMAX  = 20;
constexpr int G4_B_GMIN   = 21;
constexpr int G4_B_GMAX   = 22;

constexpr int G4_FIELD_STATS = 7;
constexpr int G4_F_POWER = 0;
constexpr int G4_F_X1    = 1;
constexpr int G4_F_X2    = 2;
constexpr int G4_F_Y1    = 3;
constexpr int G4_F_Y2    = 4;
constexpr int G4_F_FF_RE = 5;
constexpr int G4_F_FF_IM = 6;

static void g4_resize_pair(amrex::Gpu::DeviceVector<double> &dev,
                           std::vector<double> &host,
                           std::size_t n)
{
  if (dev.size() < n) { dev.resize(n); }
  if (host.size() < n) { host.resize(n); }
}

static void g4_copy_device_to_host(amrex::Gpu::DeviceVector<double> &dev,
                                   std::vector<double> &host,
                                   std::size_t n)
{
  if (n == 0) { return; }
#ifdef AMREX_USE_CUDA
  cudaMemcpyAsync(host.data(), dev.data(), n * sizeof(double),
                  cudaMemcpyDeviceToHost, amrex::Gpu::gpuStream());
#else
  amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                        dev.begin(), dev.begin() + n, host.begin());
#endif
}

__device__ inline double g4_block_reduce_sum(double val, double *sh)
{
  const int tid = threadIdx.x;
  sh[tid] = val;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) { sh[tid] += sh[tid + stride]; }
    __syncthreads();
  }
  return sh[0];
}

__device__ inline double g4_block_reduce_min(double val, double *sh)
{
  const int tid = threadIdx.x;
  sh[tid] = val;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) { sh[tid] = fmin(sh[tid], sh[tid + stride]); }
    __syncthreads();
  }
  return sh[0];
}

__device__ inline double g4_block_reduce_max(double val, double *sh)
{
  const int tid = threadIdx.x;
  sh[tid] = val;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) { sh[tid] = fmax(sh[tid], sh[tid + stride]); }
    __syncthreads();
  }
  return sh[0];
}

__global__ void g4_beam_slice_reduce_kernel(int nslice,
                                            int nharm,
                                            bool need_aux,
                                            const int *slice_offsets,
                                            const double *x,
                                            const double *y,
                                            const double *px,
                                            const double *py,
                                            const double *gamma,
                                            const double *theta,
                                            double *stats,
                                            double *b_re,
                                            double *b_im)
{
  const int is = blockIdx.x;
  if (is >= nslice) { return; }

  const int tid = threadIdx.x;
  const int start = slice_offsets[is];
  const int end = slice_offsets[is + 1];
  __shared__ double sh[G4_DIAG_THREADS];

  double g1 = 0.0, g2 = 0.0;
  double x1 = 0.0, x2 = 0.0;
  double y1 = 0.0, y2 = 0.0;
  double px1 = 0.0, px2 = 0.0;
  double py1 = 0.0, py2 = 0.0;
  double xpx = 0.0, ypy = 0.0;
  double cnt = 0.0;

  double xmin = 1.0e5, xmax = -1.0e5;
  double pxmin = 1.0e5, pxmax = -1.0e5;
  double ymin = 1.0e5, ymax = -1.0e5;
  double pymin = 1.0e5, pymax = -1.0e5;
  double gmin = 1.0e7, gmax = 1.0;

  for (int ip = start + tid; ip < end; ip += blockDim.x) {
    const double xv = x[ip];
    const double yv = y[ip];
    const double pxv = px[ip];
    const double pyv = py[ip];
    const double gv = gamma[ip];

    g1 += gv;
    g2 += gv * gv;
    x1 += xv;
    x2 += xv * xv;
    y1 += yv;
    y2 += yv * yv;
    px1 += pxv;
    px2 += pxv * pxv;
    py1 += pyv;
    py2 += pyv * pyv;
    xpx += xv * pxv;
    ypy += yv * pyv;
    cnt += 1.0;

    if (need_aux) {
      xmin = fmin(xmin, xv); xmax = fmax(xmax, xv);
      pxmin = fmin(pxmin, pxv); pxmax = fmax(pxmax, pxv);
      ymin = fmin(ymin, yv); ymax = fmax(ymax, yv);
      pymin = fmin(pymin, pyv); pymax = fmax(pymax, pyv);
      gmin = fmin(gmin, gv); gmax = fmax(gmax, gv);
    }
  }

  double *s = stats + is * G4_BEAM_STATS;
#define G4_STORE_SUM(IDX, VAL) do { double r = g4_block_reduce_sum((VAL), sh); if (tid == 0) { s[(IDX)] = r; } } while (0)
#define G4_STORE_MIN(IDX, VAL) do { double r = g4_block_reduce_min((VAL), sh); if (tid == 0) { s[(IDX)] = r; } } while (0)
#define G4_STORE_MAX(IDX, VAL) do { double r = g4_block_reduce_max((VAL), sh); if (tid == 0) { s[(IDX)] = r; } } while (0)
  G4_STORE_SUM(G4_B_G1, g1);
  G4_STORE_SUM(G4_B_G2, g2);
  G4_STORE_SUM(G4_B_X1, x1);
  G4_STORE_SUM(G4_B_X2, x2);
  G4_STORE_SUM(G4_B_Y1, y1);
  G4_STORE_SUM(G4_B_Y2, y2);
  G4_STORE_SUM(G4_B_PX1, px1);
  G4_STORE_SUM(G4_B_PX2, px2);
  G4_STORE_SUM(G4_B_PY1, py1);
  G4_STORE_SUM(G4_B_PY2, py2);
  G4_STORE_SUM(G4_B_XPX, xpx);
  G4_STORE_SUM(G4_B_YPY, ypy);
  G4_STORE_SUM(G4_B_COUNT, cnt);
  if (need_aux) {
    G4_STORE_MIN(G4_B_XMIN, xmin);
    G4_STORE_MAX(G4_B_XMAX, xmax);
    G4_STORE_MIN(G4_B_PXMIN, pxmin);
    G4_STORE_MAX(G4_B_PXMAX, pxmax);
    G4_STORE_MIN(G4_B_YMIN, ymin);
    G4_STORE_MAX(G4_B_YMAX, ymax);
    G4_STORE_MIN(G4_B_PYMIN, pymin);
    G4_STORE_MAX(G4_B_PYMAX, pymax);
    G4_STORE_MIN(G4_B_GMIN, gmin);
    G4_STORE_MAX(G4_B_GMAX, gmax);
  }
#undef G4_STORE_SUM
#undef G4_STORE_MIN
#undef G4_STORE_MAX

  for (int ih = 0; ih < nharm; ++ih) {
    double br = 0.0;
    double bi = 0.0;
    const double h = static_cast<double>(ih + 1);
    for (int ip = start + tid; ip < end; ip += blockDim.x) {
      const double ph = h * theta[ip];
      br += cos(ph);
      bi += sin(ph);
    }
    const double rr = g4_block_reduce_sum(br, sh);
    const double ii = g4_block_reduce_sum(bi, sh);
    if (tid == 0) {
      b_re[is * nharm + ih] = rr;
      b_im[is * nharm + ih] = ii;
    }
  }
}

__global__ void g4_field_slice_reduce_kernel(int nslice,
                                             int ngrid,
                                             int first,
                                             const double *field_re,
                                             const double *field_im,
                                             double *stats,
                                             double *center_re,
                                             double *center_im)
{
  const int is0 = blockIdx.x;
  if (is0 >= nslice) { return; }

  const int tid = threadIdx.x;
  const int plane = ngrid * ngrid;
  const int base = is0 * plane;
  const int is = (nslice + is0 - first) % nslice;
  const double shift = -0.5 * static_cast<double>(ngrid - 1);
  __shared__ double sh[G4_DIAG_THREADS];

  double power = 0.0;
  double x1 = 0.0, x2 = 0.0;
  double y1 = 0.0, y2 = 0.0;
  double ff_re = 0.0, ff_im = 0.0;

  for (int ixy = tid; ixy < plane; ixy += blockDim.x) {
    const int iy = ixy / ngrid;
    const int ix = ixy - iy * ngrid;
    const double re = field_re[base + ixy];
    const double im = field_im[base + ixy];
    const double wei = re * re + im * im;
    const double dx = static_cast<double>(ix) + shift;
    const double dy = static_cast<double>(iy) + shift;

    power += wei;
    x1 += dx * wei;
    x2 += dx * dx * wei;
    y1 += dy * wei;
    y2 += dy * dy * wei;
    ff_re += re;
    ff_im += im;
  }

  double *s = stats + is * G4_FIELD_STATS;
#define G4_STORE_SUM(IDX, VAL) do { double r = g4_block_reduce_sum((VAL), sh); if (tid == 0) { s[(IDX)] = r; } } while (0)
  G4_STORE_SUM(G4_F_POWER, power);
  G4_STORE_SUM(G4_F_X1, x1);
  G4_STORE_SUM(G4_F_X2, x2);
  G4_STORE_SUM(G4_F_Y1, y1);
  G4_STORE_SUM(G4_F_Y2, y2);
  G4_STORE_SUM(G4_F_FF_RE, ff_re);
  G4_STORE_SUM(G4_F_FF_IM, ff_im);
#undef G4_STORE_SUM

  if (tid == 0) {
    const int icenter = (plane - 1) / 2;
    center_re[is] = field_re[base + icenter];
    center_im[is] = field_im[base + icenter];
  }
}
#endif

} // namespace

Diagnostic::Diagnostic()
{
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_);
#ifdef GENESIS_USE_AMREX
  amrex_scratch_.reset(new DiagnosticAMReXScratch());
#endif
}

Diagnostic::~Diagnostic()
{
#ifdef FFTW
  for (auto &d: dfield) {
    if (auto *std_field = dynamic_cast<DiagField *>(d)) {
      std_field->cleanup_FFT_resources();
    }
  }
#endif
  for(auto &d: dfield) { delete d; }
  for(auto &d: dbeam) { delete d; }
}

bool Diagnostic::add_beam_diag(DiagBeamBase *pd)
{
  if(!diag_can_add) { return false; }
  dbeam.push_back(pd);
  return true;
}

bool Diagnostic::add_field_diag(DiagFieldBase *pd)
{
  if(!diag_can_add) { return false; }
  dfield.push_back(pd);
  return true;
}

void Diagnostic::init(int rank, int size, int nz_in, int ns_in, int nfld,
                      bool isTime, bool isScan, FilterDiagnostics &filter)
{
  diag_can_add = false;
#ifdef GENESIS_USE_AMREX
  filter_ = filter;
#endif
  nz = nz_in;
  ns = ns_in;
  time = isTime;
  scan = isScan;
  noff = rank * ns;
  ntotal = ns * size;

  val.clear();
  units.clear();
  single.clear();
  val.resize(3 + nfld);
  units.resize(3 + nfld);
  single.resize(3 + nfld);
  zout.resize(nz);

  for (auto group : dbeam) {
    for (auto const &tag : group->getTags(filter)) {
      units.at(2)[tag.first] = tag.second.units;
      single.at(2)[tag.first] = tag.second.global;
      int asize = ns * nz;
      if (tag.second.once) { asize /= nz; }
      if (tag.second.global) { asize /= ns; }
      val.at(2)[tag.first].resize(asize);
    }
  }

  for (int ifld = 0; ifld < nfld; ++ifld) {
    for (auto group : dfield) {
      for (auto const &tag : group->getTags(filter)) {
        if (units.at(3+ifld).find(tag.first) != units.at(3+ifld).end()) {
          if (my_rank_ == 0) {
            cout << "Diagnostic::init: warning: key " << tag.first
                 << " already existing in map (possible reason: multiple plugins with same obj_prefix)" << endl;
          }
        }
        units.at(3+ifld)[tag.first] = tag.second.units;
        single.at(3+ifld)[tag.first] = tag.second.global;
        int asize = ns * nz;
        if (tag.second.once) { asize /= nz; }
        if (tag.second.global) { asize /= ns; }
        val.at(3+ifld)[tag.first].resize(asize);
      }
    }
  }
  iz = 0;
}

void Diagnostic::addOutput(int groupID, string key, string unit, vector<double> &data)
{
  val.at(groupID)[key] = data;
  units.at(groupID)[key] = unit;
  single.at(groupID)[key] = true;
}

bool Diagnostic::writeToOutputFile(Beam *beam, vector<Field *> *field, Setup *setup, Undulator *und)
{
  diag_can_add = false;

  string rn, fnout;
  setup->getRootName(&rn);
  setup->RootName_to_FileName(&fnout, &rn);
  fnout.append(".out.h5");

  string fnmeta;
  setup->RootName_to_FileName(&fnmeta, &rn);
  fnmeta.append(".meta.h5");

  this->addOutput(0,"zplot","m", zout);
  this->addOutput(0,"z","m", und->z);
  this->addOutput(0,"dz","m", und->dz);
  this->addOutput(0,"aw"," ", und->aw);
  this->addOutput(0,"ax","m", und->ax);
  this->addOutput(0,"ay","m", und->ay);
  this->addOutput(0,"ku","m^{-1}", und->ku);
  this->addOutput(0,"kx"," ", und->kx);
  this->addOutput(0,"ky"," ", und->ky);
  this->addOutput(0,"qf","m^{-2}", und->qf);
  this->addOutput(0,"qx","m", und->qx);
  this->addOutput(0,"qy","m", und->qy);
  this->addOutput(0,"cx","rad", und->cx);
  this->addOutput(0,"cy","rad", und->cy);
  this->addOutput(0,"gradx","m^{-1}", und->gradx);
  this->addOutput(0,"grady","m^{-1}", und->grady);
  this->addOutput(0,"slippage"," ", und->slip);
  this->addOutput(0,"phaseshift","rad", und->phaseshift);
  this->addOutput(0,"chic_angle","degree", und->chic_angle);
  this->addOutput(0,"chic_lb","m", und->chic_lb);
  this->addOutput(0,"chic_ld","m", und->chic_ld);
  this->addOutput(0,"chic_lt","m", und->chic_lt);

  vector<double> global(1);
  global[0] = und->getGammaRef();
  this->addOutput(1,"gamma0","mc^2",global);
  global[0] = beam->reflength;
  this->addOutput(1,"lambdaref","m",global);
  global[0] = beam->slicelength / beam->reflength;
  this->addOutput(1,"sample"," ",global);
  global[0] = beam->slicelength * ntotal;
  this->addOutput(1,"slen","m",global);
  global[0] = beam->one4one ? 1.0 : 0.0;
  this->addOutput(1,"one4one"," ",global);
  global[0] = time ? 1.0 : 0.0;
  this->addOutput(1,"time"," ",global);
  global[0] = scan ? 1.0 : 0.0;
  this->addOutput(1,"scan"," ",global);

  global.resize(ntotal);
  for (int i = 0; i < ntotal; i++) {
    global[i] = static_cast<double>(i) * beam->slicelength;
  }
  this->addOutput(1,"s","m",global);

  const double e0 = 1239.842e-9 / beam->reflength;
  double df = e0 * beam->reflength / beam->slicelength / static_cast<double>(ntotal);
  if (ntotal == 1) { df = 0.0; }
  for (int i = 0; i < ntotal; i++) {
    global[i] = static_cast<double>(i) * df - 0.5 * df * static_cast<double>(ntotal);
  }
  this->addOutput(1,"frequency","eV",global);

  if(setup->get_do_write_outfile()) {
    Output out;
    if(!out.open(fnout, noff, ns)) {
      if(my_rank_==0) { cout << " unable to open output file" << endl; }
      return false;
    }
    out.writeMeta(und);
    out.writeGroup("Lattice", val[0], units[0], single[0]);
    out.writeGroup("Global", val[1], units[1], single[1]);
    out.writeGroup("Beam", val[2], units[2], single[2]);
    for (int i = 3; i < static_cast<int>(val.size()); i++) {
      const int h = field->at(i-3)->harm;
      char objname[30] = "Field";
      if (h != 1) { snprintf(objname, sizeof(objname), "Field%d", h); }
      out.writeGroup(objname, val[i], units[i], single[i]);
    }
    out.close();
  } else {
    if(my_rank_==0) {
      stringstream ss;
      ofstream ofs;
      ss << fnout << ".suppressed";
      ofs.open(ss.str(), ofstream::out);
      ofs.close();
      cout << " INFO: debug option to suppress writing of .out.h5 file is set" << endl;
    }
  }

  if(setup->get_write_meta_file()) {
    Output out_meta;
    if(!out_meta.open(fnmeta, noff, ns)) {
      if(my_rank_==0) { cout << " unable to open output file" << endl; }
      return false;
    }
    out_meta.writeMeta(und);
    out_meta.close();
  }
  return true;
}

void Diagnostic::calc(Beam* beam, vector<Field *> *field, double z)
{
  diag_can_add = false;
  zout[iz] = z;

#ifdef GENESIS_USE_AMREX
  if (calc_builtin_amrex(beam, field, z)) {
    iz++;
    return;
  }
#endif

  for (auto group: dbeam) {
    group->getValues(beam, val[2], iz);
  }
  for (int ifld = 0; ifld < static_cast<int>(field->size()); ifld++) {
    for (auto group: dfield) {
      group->getValues(field->at(ifld), val[3+ifld], iz);
    }
  }
  iz++;
}

#ifdef GENESIS_USE_AMREX
bool Diagnostic::calc_builtin_amrex(Beam *beam, vector<Field *> *field, double)
{
#ifdef FFTW
  if (filter_.field.fft) {
    amrex::Abort("GPU diagnostic path does not support FFT diagnostic yet");
  }
#endif

  // Fast GPU builtin path.  The AMReX port treats SoA as the source of truth;
  // callers are responsible for providing initialized, current BeamSoA/FieldSoA.
  // Do not silently unpack to the legacy AoS path here: that destroys GPU
  // residency and made auxiliary diagnostics dominate runtime.
  calc_builtin_beam_amrex(beam);

  for (int ifld = 0; ifld < static_cast<int>(field->size()); ++ifld) {
    calc_builtin_field_amrex(field->at(ifld), 3 + ifld);
  }
  return true;
}

bool Diagnostic::calc_builtin_beam_amrex(Beam *beam)
{
  Genesis4BeamSoA *soa = beam->beamSoA;
  const int nslice = soa->nslice;

  const int nharm = std::max(1, filter_.beam.harm);
  const bool exclharm = filter_.beam.exclharm;

  const std::size_t nstats = static_cast<std::size_t>(nslice) * G4_BEAM_STATS;
  const std::size_t nbunch = static_cast<std::size_t>(nslice) * nharm;
  auto& scratch = *amrex_scratch_;
  g4_resize_pair(scratch.beam_stats_dev, scratch.beam_stats_host, nstats);
  g4_resize_pair(scratch.beam_bre_dev, scratch.beam_bre_host, nbunch);
  g4_resize_pair(scratch.beam_bim_dev, scratch.beam_bim_host, nbunch);

  g4_beam_slice_reduce_kernel<<<nslice, G4_DIAG_THREADS, 0, amrex::Gpu::gpuStream()>>>(
      nslice,
      nharm,
      filter_.beam.auxiliar,
      soa->slice_offsets.data(),
      soa->x.data(),
      soa->y.data(),
      soa->px.data(),
      soa->py.data(),
      soa->gamma.data(),
      soa->theta.data(),
      scratch.beam_stats_dev.data(),
      scratch.beam_bre_dev.data(),
      scratch.beam_bim_dev.data());

  g4_copy_device_to_host(scratch.beam_stats_dev, scratch.beam_stats_host, nstats);
  g4_copy_device_to_host(scratch.beam_bre_dev, scratch.beam_bre_host, nbunch);
  g4_copy_device_to_host(scratch.beam_bim_dev, scratch.beam_bim_host, nbunch);
  amrex::Gpu::streamSynchronize();

  auto &out = val[2];
  const double *stats = scratch.beam_stats_host.data();
  const double *b_re = scratch.beam_bre_host.data();
  const double *b_im = scratch.beam_bim_host.data();

  double *p_energy = g4_value_ptr(out, "energy");
  double *p_energyspread = g4_value_ptr(out, "energyspread");
  double *p_xposition = g4_value_ptr(out, "xposition");
  double *p_xsize = g4_value_ptr(out, "xsize");
  double *p_yposition = g4_value_ptr(out, "yposition");
  double *p_ysize = g4_value_ptr(out, "ysize");
  double *p_pxposition = g4_value_ptr(out, "pxposition");
  double *p_pyposition = g4_value_ptr(out, "pyposition");
  double *p_current = g4_value_ptr(out, "current");
  double *p_emitx = g4_value_ptr(out, "emitx");
  double *p_emity = g4_value_ptr(out, "emity");
  double *p_betax = g4_value_ptr(out, "betax");
  double *p_betay = g4_value_ptr(out, "betay");
  double *p_alphax = g4_value_ptr(out, "alphax");
  double *p_alphay = g4_value_ptr(out, "alphay");
  double *p_global_energy = g4_value_ptr(out, "Global/energy");
  double *p_global_energyspread = g4_value_ptr(out, "Global/energyspread");
  double *p_global_xposition = g4_value_ptr(out, "Global/xposition");
  double *p_global_xsize = g4_value_ptr(out, "Global/xsize");
  double *p_global_yposition = g4_value_ptr(out, "Global/yposition");
  double *p_global_ysize = g4_value_ptr(out, "Global/ysize");

  double *p_efield = g4_value_ptr(out, "efield");
  double *p_wakefield = g4_value_ptr(out, "wakefield");
  double *p_lscfield = g4_value_ptr(out, "LSCfield");
  double *p_sscfield = g4_value_ptr(out, "SSCfield");
  double *p_xmin = g4_value_ptr(out, "xmin");
  double *p_xmax = g4_value_ptr(out, "xmax");
  double *p_pxmin = g4_value_ptr(out, "pxmin");
  double *p_pxmax = g4_value_ptr(out, "pxmax");
  double *p_ymin = g4_value_ptr(out, "ymin");
  double *p_ymax = g4_value_ptr(out, "ymax");
  double *p_pymin = g4_value_ptr(out, "pymin");
  double *p_pymax = g4_value_ptr(out, "pymax");
  double *p_emin = g4_value_ptr(out, "emin");
  double *p_emax = g4_value_ptr(out, "emax");

  std::vector<double*> p_bunching(static_cast<std::size_t>(nharm), nullptr);
  std::vector<double*> p_bunchingphase(static_cast<std::size_t>(nharm), nullptr);
  p_bunching[0] = g4_value_ptr(out, "bunching");
  p_bunchingphase[0] = g4_value_ptr(out, "bunchingphase");
  char hname[100];
  if (exclharm && nharm > 1) {
    snprintf(hname, sizeof(hname), "bunching%d", nharm);
    p_bunching[static_cast<std::size_t>(nharm - 1)] = g4_value_ptr(out, hname);
    snprintf(hname, sizeof(hname), "bunchingphase%d", nharm);
    p_bunchingphase[static_cast<std::size_t>(nharm - 1)] = g4_value_ptr(out, hname);
  } else {
    for (int ih = 1; ih < nharm; ++ih) {
      snprintf(hname, sizeof(hname), "bunching%d", ih + 1);
      p_bunching[static_cast<std::size_t>(ih)] = g4_value_ptr(out, hname);
      snprintf(hname, sizeof(hname), "bunchingphase%d", ih + 1);
      p_bunchingphase[static_cast<std::size_t>(ih)] = g4_value_ptr(out, hname);
    }
  }

  double g_cur = 0.0;
  double g_g1 = 0.0;
  double g_g2 = 0.0;
  double g_x1 = 0.0;
  double g_x2 = 0.0;
  double g_y1 = 0.0;
  double g_y2 = 0.0;

  for (int is = 0; is < nslice; ++is) {
    const double *s = stats + is * G4_BEAM_STATS;
    const double count = s[G4_B_COUNT];
    const double norm = (count > 0.0) ? 1.0 / count : 1.0;

    const double x1 = s[G4_B_X1] * norm;
    const double x2 = s[G4_B_X2] * norm;
    const double y1 = s[G4_B_Y1] * norm;
    const double y2 = s[G4_B_Y2] * norm;
    const double px1 = s[G4_B_PX1] * norm;
    const double px2 = s[G4_B_PX2] * norm;
    const double py1 = s[G4_B_PY1] * norm;
    const double py2 = s[G4_B_PY2] * norm;
    const double g1 = s[G4_B_G1] * norm;
    const double g2 = s[G4_B_G2] * norm;
    const double xpx = s[G4_B_XPX] * norm;
    const double ypy = s[G4_B_YPY] * norm;

    const unsigned long idx = static_cast<unsigned long>(iz * nslice + is);

    if (filter_.beam.energy) {
      g4_store_ptr(p_energy, idx, g1);
      g4_store_ptr(p_energyspread, idx, g4_safe_sqrt_var(g2 - g1 * g1));
    }
    if (filter_.beam.spatial) {
      g4_store_ptr(p_xposition, idx, x1);
      g4_store_ptr(p_xsize, idx, g4_safe_sqrt_var(x2 - x1 * x1));
      g4_store_ptr(p_yposition, idx, y1);
      g4_store_ptr(p_ysize, idx, g4_safe_sqrt_var(y2 - y1 * y1));
      g4_store_ptr(p_pxposition, idx, px1);
      g4_store_ptr(p_pyposition, idx, py1);
    }

    auto store_harm = [&](int ih) {
      const double bre = b_re[is * nharm + ih] * norm;
      const double bim = b_im[is * nharm + ih] * norm;
      const std::size_t k = static_cast<std::size_t>(ih);
      g4_store_ptr(p_bunching[k], idx, hypot(bre, bim));
      g4_store_ptr(p_bunchingphase[k], idx, atan2(bim, bre));
    };
    store_harm(0);
    if (exclharm && nharm > 1) {
      store_harm(nharm - 1);
    } else {
      for (int ih = 1; ih < nharm; ++ih) { store_harm(ih); }
    }

    if (filter_.beam.auxiliar) {
      const double eloss = (is < static_cast<int>(beam->eloss.size())) ? beam->eloss[is] : 0.0;
      const double longESC = (is < static_cast<int>(beam->longESC.size())) ? beam->longESC[is] : 0.0;
      g4_store_ptr(p_efield, idx, eloss + longESC);
      g4_store_ptr(p_wakefield, idx, eloss);
      g4_store_ptr(p_lscfield, idx, longESC);
      g4_store_ptr(p_sscfield, idx, beam->getSCField(is));
      g4_store_ptr(p_xmin, idx, s[G4_B_XMIN]);
      g4_store_ptr(p_xmax, idx, s[G4_B_XMAX]);
      g4_store_ptr(p_pxmin, idx, s[G4_B_PXMIN]);
      g4_store_ptr(p_pxmax, idx, s[G4_B_PXMAX]);
      g4_store_ptr(p_ymin, idx, s[G4_B_YMIN]);
      g4_store_ptr(p_ymax, idx, s[G4_B_YMAX]);
      g4_store_ptr(p_pymin, idx, s[G4_B_PYMIN]);
      g4_store_ptr(p_pymax, idx, s[G4_B_PYMAX]);
      g4_store_ptr(p_emin, idx, s[G4_B_GMIN]);
      g4_store_ptr(p_emax, idx, s[G4_B_GMAX]);
    }

    if (filter_.beam.current || iz == 0) {
      const double cur = (is < static_cast<int>(beam->current.size())) ? beam->current[is] : 0.0;
      g4_store_ptr(p_current, idx, cur);
    }

    if (filter_.beam.twiss || iz == 0) {
      const double ex = sqrt(fabs((x2 - x1 * x1) * (px2 - px1 * px1) - (xpx - x1 * px1) * (xpx - x1 * px1)));
      const double ey = sqrt(fabs((y2 - y1 * y1) * (py2 - py1 * py1) - (ypy - y1 * py1) * (ypy - y1 * py1)));
      g4_store_ptr(p_emitx, idx, ex);
      g4_store_ptr(p_emity, idx, ey);
      g4_store_ptr(p_betax, idx, (ex > 0.0) ? (x2 - x1 * x1) / ex * g1 : 0.0);
      g4_store_ptr(p_betay, idx, (ey > 0.0) ? (y2 - y1 * y1) / ey * g1 : 0.0);
      g4_store_ptr(p_alphax, idx, (ex > 0.0) ? -(xpx - x1 * px1) / ex : 0.0);
      g4_store_ptr(p_alphay, idx, (ey > 0.0) ? -(ypy - y1 * py1) / ey : 0.0);
    }

    if (filter_.beam.global) {
      const double cur = (is < static_cast<int>(beam->current.size())) ? beam->current[is] : 0.0;
      g_cur += cur;
      g_g1 += cur * g1;
      g_g2 += cur * g2;
      g_x1 += cur * x1;
      g_x2 += cur * x2;
      g_y1 += cur * y1;
      g_y2 += cur * y2;
    }
  }

  if (filter_.beam.global) {
    double gvals[7] = {g_cur, g_g1, g_g2, g_x1, g_x2, g_y1, g_y2};
    g4_allreduce_sum_inplace(gvals, 7);
    g_cur = gvals[0]; g_g1 = gvals[1]; g_g2 = gvals[2];
    g_x1 = gvals[3]; g_x2 = gvals[4]; g_y1 = gvals[5]; g_y2 = gvals[6];
    const double norm = (g_cur > 0.0) ? 1.0 / g_cur : 1.0;
    g_g1 *= norm; g_g2 *= norm; g_x1 *= norm; g_x2 *= norm; g_y1 *= norm; g_y2 *= norm;
    if (filter_.beam.energy) {
      g4_store_ptr(p_global_energy, iz, g_g1);
      g4_store_ptr(p_global_energyspread, iz, g4_safe_sqrt_var(g_g2 - g_g1 * g_g1));
    }
    if (filter_.beam.spatial) {
      g4_store_ptr(p_global_xposition, iz, g_x1);
      g4_store_ptr(p_global_xsize, iz, g4_safe_sqrt_var(g_x2 - g_x1 * g_x1));
      g4_store_ptr(p_global_yposition, iz, g_y1);
      g4_store_ptr(p_global_ysize, iz, g4_safe_sqrt_var(g_y2 - g_y1 * g_y1));
    }
  }

  return true;
}

bool Diagnostic::calc_builtin_field_amrex(Field *field, int groupID)
{
  Genesis4FieldSoA *soa = field->fieldSoA;
  const int nslice = soa->nslice;
  const int ngrid = soa->ngrid;

  const std::size_t nstats = static_cast<std::size_t>(nslice) * G4_FIELD_STATS;
  const std::size_t ncenter = static_cast<std::size_t>(nslice);
  auto& scratch = *amrex_scratch_;
  g4_resize_pair(scratch.field_stats_dev, scratch.field_stats_host, nstats);
  g4_resize_pair(scratch.field_center_re_dev, scratch.field_center_re_host, ncenter);
  g4_resize_pair(scratch.field_center_im_dev, scratch.field_center_im_host, ncenter);

  g4_field_slice_reduce_kernel<<<nslice, G4_DIAG_THREADS, 0, amrex::Gpu::gpuStream()>>>(
      nslice,
      ngrid,
      field->first,
      soa->field_re.data(),
      soa->field_im.data(),
      scratch.field_stats_dev.data(),
      scratch.field_center_re_dev.data(),
      scratch.field_center_im_dev.data());

  g4_copy_device_to_host(scratch.field_stats_dev, scratch.field_stats_host, nstats);
  g4_copy_device_to_host(scratch.field_center_re_dev, scratch.field_center_re_host, ncenter);
  g4_copy_device_to_host(scratch.field_center_im_dev, scratch.field_center_im_host, ncenter);
  amrex::Gpu::streamSynchronize();

  auto &out = val[groupID];
  const double *stats = scratch.field_stats_host.data();
  const double *center_re = scratch.field_center_re_host.data();
  const double *center_im = scratch.field_center_im_host.data();

  double *p_power = g4_value_ptr(out, "power");
  double *p_xposition = g4_value_ptr(out, "xposition");
  double *p_xsize = g4_value_ptr(out, "xsize");
  double *p_yposition = g4_value_ptr(out, "yposition");
  double *p_ysize = g4_value_ptr(out, "ysize");
  double *p_intensity_nf = g4_value_ptr(out, "intensity-nearfield");
  double *p_phase_nf = g4_value_ptr(out, "phase-nearfield");
  double *p_intensity_ff = g4_value_ptr(out, "intensity-farfield");
  double *p_phase_ff = g4_value_ptr(out, "phase-farfield");
  double *p_global_energy = g4_value_ptr(out, "Global/energy");
  double *p_global_xposition = g4_value_ptr(out, "Global/xposition");
  double *p_global_xsize = g4_value_ptr(out, "Global/xsize");
  double *p_global_yposition = g4_value_ptr(out, "Global/yposition");
  double *p_global_ysize = g4_value_ptr(out, "Global/ysize");
  double *p_global_intensity_nf = g4_value_ptr(out, "Global/intensity-nearfield");
  double *p_global_intensity_ff = g4_value_ptr(out, "Global/intensity-farfield");
  double *p_dgrid = g4_value_ptr(out, "dgrid");
  double *p_gridspacing = g4_value_ptr(out, "gridspacing");
  double *p_ngrid = g4_value_ptr(out, "ngrid");

  const double ks = 4.0 * asin(1.0) / field->xlambda;
  const double scl = field->dgrid * eev / ks;
  double g_pow = 0.0;
  double g_x1 = 0.0;
  double g_x2 = 0.0;
  double g_y1 = 0.0;
  double g_y2 = 0.0;
  double g_ff = 0.0;
  double g_inten = 0.0;

  for (int is = 0; is < nslice; ++is) {
    const double *s = stats + is * G4_FIELD_STATS;
    double power_raw = s[G4_F_POWER];
    double x1 = s[G4_F_X1];
    double x2 = s[G4_F_X2];
    double y1 = s[G4_F_Y1];
    double y2 = s[G4_F_Y2];
    const double ff_re = s[G4_F_FF_RE];
    const double ff_im = s[G4_F_FF_IM];

    if (filter_.field.global) {
      g_pow += power_raw;
      g_x1 += x1;
      g_x2 += x2;
      g_y1 += y1;
      g_y2 += y2;
    }

    if (power_raw > 0.0) {
      x1 /= power_raw;
      x2 /= power_raw;
      y1 /= power_raw;
      y2 /= power_raw;
    }
    x2 = sqrt(fabs(x2 - x1 * x1));
    y2 = sqrt(fabs(y2 - y1 * y1));

    const double cre = center_re[is];
    const double cim = center_im[is];
    double inten = cre * cre + cim * cim;
    const double intenphi = atan2(cim, cre);
    double farfield = ff_re * ff_re + ff_im * ff_im;
    const double farfieldphi = atan2(ff_im, ff_re);
    farfield *= field->dgrid * field->dgrid;

    if (filter_.field.global) {
      g_ff += farfield;
      g_inten += inten;
    }

    const double power = power_raw * scl * scl / vacimp;
    x1 *= field->dgrid;
    x2 *= field->dgrid;
    y1 *= field->dgrid;
    y2 *= field->dgrid;
    inten *= eev * eev / ks / ks / vacimp;

    const unsigned long idx = static_cast<unsigned long>(iz * nslice + is);
    g4_store_ptr(p_power, idx, power);
    if (filter_.field.spatial) {
      g4_store_ptr(p_xposition, idx, x1);
      g4_store_ptr(p_xsize, idx, x2);
      g4_store_ptr(p_yposition, idx, y1);
      g4_store_ptr(p_ysize, idx, y2);
    }
    if (filter_.field.intensity) {
      g4_store_ptr(p_intensity_nf, idx, inten);
      g4_store_ptr(p_phase_nf, idx, intenphi);
      g4_store_ptr(p_intensity_ff, idx, farfield);
      g4_store_ptr(p_phase_ff, idx, farfieldphi);
    }
  }

  if (filter_.field.global) {
    int mpi_size = 1;
    if (!MPISingle) { MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); }
    double gvals[7] = {g_pow, g_x1, g_x2, g_y1, g_y2, g_ff, g_inten};
    g4_allreduce_sum_inplace(gvals, 7);
    g_pow = gvals[0]; g_x1 = gvals[1]; g_x2 = gvals[2];
    g_y1 = gvals[3]; g_y2 = gvals[4]; g_ff = gvals[5]; g_inten = gvals[6];

    const double norm = (g_pow > 0.0) ? 1.0 / g_pow : 1.0;
    g_x1 *= norm;
    g_x2 *= norm;
    g_y1 *= norm;
    g_y2 *= norm;
    g_x2 = sqrt(fabs(g_x2 - g_x1 * g_x1)) * field->dgrid;
    g_y2 = sqrt(fabs(g_y2 - g_y1 * g_y1)) * field->dgrid;
    g_x1 *= field->dgrid;
    g_y1 *= field->dgrid;
    const double avg_norm = 1.0 / static_cast<double>(mpi_size * nslice);
    g_inten *= avg_norm;
    g_ff *= avg_norm;

    g4_store_ptr(p_global_energy, iz, g_pow * scl * scl / vacimp * field->slicelength / 299792458.0);
    if (filter_.field.spatial) {
      g4_store_ptr(p_global_xposition, iz, g_x1);
      g4_store_ptr(p_global_xsize, iz, g_x2);
      g4_store_ptr(p_global_yposition, iz, g_y1);
      g4_store_ptr(p_global_ysize, iz, g_y2);
    }
    if (filter_.field.intensity) {
      g4_store_ptr(p_global_intensity_nf, iz, g_inten * eev * eev / ks / ks / vacimp);
      g4_store_ptr(p_global_intensity_ff, iz, g_ff);
    }
  }

  if (iz == 0) {
    g4_store_ptr(p_dgrid, 0, field->gridmax);
    g4_store_ptr(p_gridspacing, 0, field->dgrid);
    g4_store_ptr(p_ngrid, 0, static_cast<double>(ngrid));
  }

  return true;
}

#endif // GENESIS_USE_AMREX

// -----------------------------------------------------------
// actual implementation of diagnostic calculation - beam
map<string,OutputInfo> DiagBeam::getTags(FilterDiagnostics &filter_in)
{
  tags.clear();
  filter.clear();
  nharm = filter_in.beam.harm;
  exclharm = filter_in.beam.exclharm;
  global = filter_in.beam.global;

  if (filter_in.beam.energy) {
    filter["energy"] = true;
    tags["energy"] = {false, false, "mc^2"};
    tags["energyspread"] = {false, false, "mc^2"};
    if (global) {
      tags["Global/energy"] = {true, false, "mc^2"};
      tags["Global/energyspread"] = {true, false, "mc^2"};
    }
  }
  if (filter_in.beam.spatial) {
    filter["spatial"] = true;
    tags["xposition"] = {false, false, "m"};
    tags["xsize"] = {false, false, "m"};
    tags["yposition"] = {false, false, "m"};
    tags["ysize"] = {false, false, "m"};
    tags["pxposition"] = {false, false, "rad"};
    tags["pyposition"] = {false, false, "rad"};
    if (global) {
      tags["Global/xposition"] = {true, false, "m"};
      tags["Global/xsize"] = {true, false, "m"};
      tags["Global/yposition"] = {true, false, "m"};
      tags["Global/ysize"] = {true, false, "m"};
    }
  }

  tags["bunching"] = {false, false, " "};
  tags["bunchingphase"] = {false, false, "rad"};
  char buff[30];
  if (exclharm && nharm > 1) {
    snprintf(buff, sizeof(buff), "bunching%d", nharm);
    tags[buff] = {false, false, " "};
    snprintf(buff, sizeof(buff), "bunchingphase%d", nharm);
    tags[buff] = {false, false, "rad"};
  } else {
    for (unsigned int iharm = 1; iharm < nharm; iharm++) {
      snprintf(buff, sizeof(buff), "bunching%d", iharm + 1);
      tags[buff] = {false, false, " "};
      snprintf(buff, sizeof(buff), "bunchingphase%d", iharm + 1);
      tags[buff] = {false, false, "rad"};
    }
  }

  if (filter_in.beam.auxiliar) {
    filter["aux"] = true;
    tags["efield"] = {false, false, "eV/m"};
    tags["wakefield"] = {false, false, "eV/m"};
    tags["LSCfield"] = {false, false, "eV/m"};
    tags["SSCfield"] = {false, false, "eV/m"};
    tags["xmin"] = {false, false, "m"};
    tags["xmax"] = {false, false, "m"};
    tags["pxmin"] = {false, false, "rad"};
    tags["pxmax"] = {false, false, "rad"};
    tags["ymin"] = {false, false, "m"};
    tags["ymax"] = {false, false, "m"};
    tags["pymin"] = {false, false, "rad"};
    tags["pymax"] = {false, false, "rad"};
    tags["emin"] = {false, false, "mc^2"};
    tags["emax"] = {false, false, "mc^2"};
  }

  if (filter_in.beam.current) {
    tags["current"] = {false, false, "A"};
  } else {
    tags["current"] = {false, true, "A"};
  }

  if (filter_in.beam.twiss) {
    tags["emitx"] = {false, false, "m"};
    tags["emity"] = {false, false, "m"};
    tags["betax"] = {false, false, "m"};
    tags["betay"] = {false, false, "m"};
    tags["alphax"] = {false, false, "rad"};
    tags["alphay"] = {false, false, "rad"};
  } else {
    tags["emitx"] = {false, true, "m"};
    tags["emity"] = {false, true, "m"};
    tags["betax"] = {false, true, "m"};
    tags["betay"] = {false, true, "m"};
    tags["alphax"] = {false, true, "rad"};
    tags["alphay"] = {false, true, "rad"};
  }
  return tags;
}

void DiagBeam::getValues(Beam *beam, map<string, vector<double> > &val, int iz)
{
  const int ns = beam->beam.size();
  const unsigned int nh = std::max(1u, nharm);
  int is = 0;

  double g_cur = 0.0, g_g1 = 0.0, g_g2 = 0.0, g_x1 = 0.0, g_x2 = 0.0, g_y1 = 0.0, g_y2 = 0.0;

  for (auto const &slice: beam->beam) {
    double g1 = 0.0, g2 = 0.0, x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    double px1 = 0.0, py1 = 0.0, px2 = 0.0, py2 = 0.0, xpx = 0.0, ypy = 0.0;
    double xmin = 1e5, xmax = -1e5, pxmin = 1e5, pxmax = -1e5;
    double ymin = 1e5, ymax = -1e5, pymin = 1e5, pymax = -1e5;
    double gmin = 1e7, gmax = 1;
    vector<complex<double> > b(nh, complex<double>(0.0, 0.0));

    for (auto const &par: slice) {
      x1 += par.x; x2 += par.x * par.x;
      y1 += par.y; y2 += par.y * par.y;
      g1 += par.gamma; g2 += par.gamma * par.gamma;
      px1 += par.px; py1 += par.py;
      px2 += par.px * par.px; py2 += par.py * par.py;
      xpx += par.x * par.px; ypy += par.y * par.py;
      for (unsigned int ih = 0; ih < nh; ++ih) {
        b[ih] += complex<double>(cos(static_cast<double>(ih + 1) * par.theta),
                                 sin(static_cast<double>(ih + 1) * par.theta));
      }
      xmin = std::min(xmin, par.x); xmax = std::max(xmax, par.x);
      pxmin = std::min(pxmin, par.px); pxmax = std::max(pxmax, par.px);
      ymin = std::min(ymin, par.y); ymax = std::max(ymax, par.y);
      pymin = std::min(pymin, par.py); pymax = std::max(pymax, par.py);
      gmin = std::min(gmin, par.gamma); gmax = std::max(gmax, par.gamma);
    }

    const double norm = slice.empty() ? 1.0 : 1.0 / static_cast<double>(slice.size());
    x1 *= norm; x2 *= norm; y1 *= norm; y2 *= norm;
    px1 *= norm; px2 *= norm; py1 *= norm; py2 *= norm;
    g1 *= norm; g2 *= norm; xpx *= norm; ypy *= norm;
    for (auto &bi: b) { bi *= norm; }

    const unsigned long idx = static_cast<unsigned long>(iz * ns + is);
    if (filter["energy"]) {
      storeValue(val, "energy", idx, g1);
      storeValue(val, "energyspread", idx, sqrt(fabs(g2 - g1 * g1)));
    }
    if (filter["spatial"]) {
      storeValue(val, "xposition", idx, x1);
      storeValue(val, "xsize", idx, sqrt(fabs(x2 - x1 * x1)));
      storeValue(val, "yposition", idx, y1);
      storeValue(val, "ysize", idx, sqrt(fabs(y2 - y1 * y1)));
      storeValue(val, "pxposition", idx, px1);
      storeValue(val, "pyposition", idx, py1);
    }
    storeValue(val, "bunching", idx, abs(b[0]));
    storeValue(val, "bunchingphase", idx, atan2(b[0].imag(), b[0].real()));
    char buff[100];
    if (exclharm && nh > 1) {
      snprintf(buff, sizeof(buff), "bunching%d", nh);
      storeValue(val, buff, idx, abs(b[nh-1]));
      snprintf(buff, sizeof(buff), "bunchingphase%d", nh);
      storeValue(val, buff, idx, atan2(b[nh-1].imag(), b[nh-1].real()));
    } else {
      for (unsigned int ih = 1; ih < nh; ++ih) {
        snprintf(buff, sizeof(buff), "bunching%d", ih + 1);
        storeValue(val, buff, idx, abs(b[ih]));
        snprintf(buff, sizeof(buff), "bunchingphase%d", ih + 1);
        storeValue(val, buff, idx, atan2(b[ih].imag(), b[ih].real()));
      }
    }

    if (filter["aux"]) {
      storeValue(val, "efield", idx, beam->eloss[is] + beam->longESC[is]);
      storeValue(val, "wakefield", idx, beam->eloss[is]);
      storeValue(val, "LSCfield", idx, beam->longESC[is]);
      storeValue(val, "SSCfield", idx, beam->getSCField(is));
      storeValue(val, "xmin", idx, xmin); storeValue(val, "xmax", idx, xmax);
      storeValue(val, "pxmin", idx, pxmin); storeValue(val, "pxmax", idx, pxmax);
      storeValue(val, "ymin", idx, ymin); storeValue(val, "ymax", idx, ymax);
      storeValue(val, "pymin", idx, pymin); storeValue(val, "pymax", idx, pymax);
      storeValue(val, "emin", idx, gmin); storeValue(val, "emax", idx, gmax);
    }

    if (tags["current"].once) {
      if (iz == 0) { storeValue(val, "current", idx, beam->current[is]); }
    } else {
      storeValue(val, "current", idx, beam->current[is]);
    }

    const double ex = sqrt(fabs((x2 - x1 * x1) * (px2 - px1 * px1) - (xpx - x1 * px1) * (xpx - x1 * px1)));
    const double ey = sqrt(fabs((y2 - y1 * y1) * (py2 - py1 * py1) - (ypy - y1 * py1) * (ypy - y1 * py1)));
    if (!tags["emitx"].once || iz == 0) {
      storeValue(val, "emitx", idx, ex); storeValue(val, "emity", idx, ey);
      storeValue(val, "betax", idx, (ex > 0.0) ? (x2 - x1 * x1) / ex * g1 : 0.0);
      storeValue(val, "betay", idx, (ey > 0.0) ? (y2 - y1 * y1) / ey * g1 : 0.0);
      storeValue(val, "alphax", idx, (ex > 0.0) ? -(xpx - x1 * px1) / ex : 0.0);
      storeValue(val, "alphay", idx, (ey > 0.0) ? -(ypy - y1 * py1) / ey : 0.0);
    }

    if (global) {
      g_cur += beam->current[is];
      g_g1 += beam->current[is] * g1;
      g_g2 += beam->current[is] * g2;
      g_x1 += beam->current[is] * x1;
      g_x2 += beam->current[is] * x2;
      g_y1 += beam->current[is] * y1;
      g_y2 += beam->current[is] * y2;
    }
    is++;
  }

  if (global) {
    double gvals[7] = {g_cur, g_g1, g_g2, g_x1, g_x2, g_y1, g_y2};
    g4_allreduce_sum_inplace(gvals, 7);
    g_cur = gvals[0]; g_g1 = gvals[1]; g_g2 = gvals[2];
    g_x1 = gvals[3]; g_x2 = gvals[4]; g_y1 = gvals[5]; g_y2 = gvals[6];
    const double norm = (g_cur > 0.0) ? 1.0 / g_cur : 1.0;
    g_g1 *= norm; g_g2 *= norm; g_x1 *= norm; g_x2 *= norm; g_y1 *= norm; g_y2 *= norm;
    if (filter["energy"]) {
      storeValue(val, "Global/energy", iz, g_g1);
      storeValue(val, "Global/energyspread", iz, sqrt(fabs(g_g2 - g_g1 * g_g1)));
    }
    if (filter["spatial"]) {
      storeValue(val, "Global/xposition", iz, g_x1);
      storeValue(val, "Global/xsize", iz, sqrt(fabs(g_x2 - g_x1 * g_x1)));
      storeValue(val, "Global/yposition", iz, g_y1);
      storeValue(val, "Global/ysize", iz, sqrt(fabs(g_y2 - g_y1 * g_y1)));
    }
  }
}

#ifdef FFTW
void DiagField::cleanup_FFT_resources(void)
{
  for(auto &[k,obj]: fftobj) { delete obj; }
  fftobj.clear();
}

int DiagField::obtain_FFT_resources(int ngrid, complex<double> **in, complex<double> **out, fftw_plan *pp)
{
  auto ele = fftobj.find(ngrid);
  if (ele == fftobj.end()) {
    FFTObj *n = new FFTObj(ngrid);
    fftobj.insert({ngrid, n});
    ele = fftobj.find(ngrid);
  }
  *in = ele->second->in_;
  *out = ele->second->out_;
  *pp = ele->second->p_;
  return 0;
}
#endif

map<string,OutputInfo> DiagField::getTags(FilterDiagnostics &filter_in)
{
  tags.clear();
  filter.clear();
  global = filter_in.field.global;

  tags["power"] = {false, false, "W"};
  if (global) { tags["Global/energy"] = {true, false, "J"}; }

  if (filter_in.field.spatial) {
    filter["spatial"] = true;
    tags["xposition"] = {false, false, "m"};
    tags["xsize"] = {false, false, "m"};
    tags["yposition"] = {false, false, "m"};
    tags["ysize"] = {false, false, "m"};
    if (global) {
      tags["Global/xposition"] = {true, false, "m"};
      tags["Global/xsize"] = {true, false, "m"};
      tags["Global/yposition"] = {true, false, "m"};
      tags["Global/ysize"] = {true, false, "m"};
    }
  }

  if (filter_in.field.intensity) {
    filter["intensity"] = true;
    tags["intensity-nearfield"] = {false, false, "W/m^2"};
    tags["phase-nearfield"] = {false, false, "rad"};
    tags["intensity-farfield"] = {false, false, "W/rad^2"};
    tags["phase-farfield"] = {false, false, "rad"};
    if (global) {
      tags["Global/intensity-nearfield"] = {true, false, "W/m^2"};
      tags["Global/intensity-farfield"] = {true, false, "W/rad^2"};
    }
  }

#ifdef FFTW
  if (filter_in.field.fft) {
    filter["fft"] = true;
    tags["xpointing"] = {false, false, "rad"};
    tags["xdivergence"] = {false, false, "rad"};
    tags["ypointing"] = {false, false, "rad"};
    tags["ydivergence"] = {false, false, "rad"};
    if (global) {
      tags["Global/xpointing"] = {true, false, "rad"};
      tags["Global/xdivergence"] = {true, false, "rad"};
      tags["Global/ypointing"] = {true, false, "rad"};
      tags["Global/ydivergence"] = {true, false, "rad"};
    }
  }
#endif

  // some basic parameter output
  tags["gridspacing"] = {true, true, "m"};
  tags["dgrid"] = {true, true, "m"};
  tags["ngrid"] = {true, true, " "};

  return tags;
}

void DiagField::getValues(Field *field, map<string, vector<double> > &val, int iz)
{
  const int ns = field->field.size();
  const int ngrid = field->ngrid;
  const double ks = 4.0 * asin(1.0) / field->xlambda;
  const double scl = field->dgrid * eev / ks;
  const double shift = -0.5 * static_cast<double>(ngrid - 1);

  double g_pow = 0.0, g_x1 = 0.0, g_x2 = 0.0, g_y1 = 0.0, g_y2 = 0.0;
  double g_ff = 0.0, g_inten = 0.0;

  int is0 = 0;
  for (auto const &slice : field->field) {
    const int is = (ns + is0 - field->first) % ns;
    double power = 0.0, x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    complex<double> ff(0.0, 0.0);

    for (int iy = 0; iy < ngrid; iy++) {
      const double dy = static_cast<double>(iy) + shift;
      for (int ix = 0; ix < ngrid; ix++) {
        const double dx = static_cast<double>(ix) + shift;
        const int i = iy * ngrid + ix;
        const complex<double> loc = slice.at(i);
        const double wei = loc.real() * loc.real() + loc.imag() * loc.imag();
        ff += loc;
        power += wei;
        x1 += dx * wei;
        x2 += dx * dx * wei;
        y1 += dy * wei;
        y2 += dy * dy * wei;
      }
    }

    if (global) {
      g_pow += power;
      g_x1 += x1;
      g_x2 += x2;
      g_y1 += y1;
      g_y2 += y2;
    }

    if (power > 0.0) {
      x1 /= power; x2 /= power; y1 /= power; y2 /= power;
    }
    x2 = sqrt(fabs(x2 - x1 * x1));
    y2 = sqrt(fabs(y2 - y1 * y1));

    const int ic = (ngrid * ngrid - 1) / 2;
    complex<double> loc = slice.at(ic);
    double inten = loc.real() * loc.real() + loc.imag() * loc.imag();
    const double intenphi = atan2(loc.imag(), loc.real());
    double farfield = ff.real() * ff.real() + ff.imag() * ff.imag();
    const double farfieldphi = atan2(ff.imag(), ff.real());
    farfield *= field->dgrid * field->dgrid;
    if (global) {
      g_ff += farfield;
      g_inten += inten;
    }

    power *= scl * scl / vacimp;
    x1 *= field->dgrid; x2 *= field->dgrid; y1 *= field->dgrid; y2 *= field->dgrid;
    inten *= eev * eev / ks / ks / vacimp;

    const unsigned long idx = static_cast<unsigned long>(iz * ns + is);
    storeValue(val, "power", idx, power);
    if (filter["spatial"]) {
      storeValue(val, "xposition", idx, x1);
      storeValue(val, "xsize", idx, x2);
      storeValue(val, "yposition", idx, y1);
      storeValue(val, "ysize", idx, y2);
    }
    if (filter["intensity"]) {
      storeValue(val, "intensity-nearfield", idx, inten);
      storeValue(val, "phase-nearfield", idx, intenphi);
      storeValue(val, "intensity-farfield", idx, farfield);
      storeValue(val, "phase-farfield", idx, farfieldphi);
    }
    is0++;
  }

  if (global) {
    int size = 1;
    if (!MPISingle) { MPI_Comm_size(MPI_COMM_WORLD, &size); }
    double gvals[7] = {g_pow, g_x1, g_x2, g_y1, g_y2, g_ff, g_inten};
    g4_allreduce_sum_inplace(gvals, 7);
    g_pow = gvals[0]; g_x1 = gvals[1]; g_x2 = gvals[2];
    g_y1 = gvals[3]; g_y2 = gvals[4]; g_ff = gvals[5]; g_inten = gvals[6];
    const double norm = (g_pow > 0.0) ? 1.0 / g_pow : 1.0;
    g_x1 *= norm; g_x2 *= norm; g_y1 *= norm; g_y2 *= norm;
    g_x2 = sqrt(fabs(g_x2 - g_x1 * g_x1)) * field->dgrid;
    g_y2 = sqrt(fabs(g_y2 - g_y1 * g_y1)) * field->dgrid;
    g_x1 *= field->dgrid;
    g_y1 *= field->dgrid;
    const double avg_norm = 1.0 / static_cast<double>(size * ns);
    g_inten *= avg_norm;
    g_ff *= avg_norm;

    storeValue(val, "Global/energy", iz, g_pow * scl * scl / vacimp * field->slicelength / 299792458.0);
    if (filter["spatial"]) {
      storeValue(val, "Global/xposition", iz, g_x1);
      storeValue(val, "Global/xsize", iz, g_x2);
      storeValue(val, "Global/yposition", iz, g_y1);
      storeValue(val, "Global/ysize", iz, g_y2);
    }
    if (filter["intensity"]) {
      storeValue(val, "Global/intensity-nearfield", iz, g_inten * eev * eev / ks / ks / vacimp);
      storeValue(val, "Global/intensity-farfield", iz, g_ff);
    }
  }

  if (iz == 0) {
    storeValue(val, "dgrid", 0, field->gridmax);
    storeValue(val, "gridspacing", 0, field->dgrid);
    storeValue(val, "ngrid", 0, static_cast<double>(ngrid));
  }
}
