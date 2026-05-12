#include "BeamSolver.h"

#include "Field.h"
#include "Beam.h"

#ifdef GENESIS_USE_AMREX
#include "Genesis4BeamSoA.h"
#include "Genesis4FieldSoA.h"
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <cmath>
#include <cstddef>
#endif

#ifdef GENESIS_USE_AMREX
namespace {
constexpr int kMaxBeamSolverFields = 8;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void genesis4_beamsolver_ode_gpu(double tgam,
                                  double tthet,
                                  double btpar,
                                  double ez,
                                  double xks,
                                  double xku,
                                  int nfield,
                                  const double* rharm,
                                  const double* rpart_re,
                                  const double* rpart_im,
                                  double& k2gg,
                                  double& k2pp) noexcept
{
  const double ztemp1 = -2.0 / xks;
  double ctmp_re = 0.0;
  double ctmp_im = 0.0;

  for (int i = 0; i < nfield; ++i) {
    const double arg = rharm[i] * tthet;
    const double c = cos(arg);
    const double s = sin(arg);
    const double rr = rpart_re[i];
    const double ri = rpart_im[i];

    // (rr + i*ri) * (cos(arg) - i*sin(arg))
    ctmp_re += rr * c + ri * s;
    ctmp_im += ri * c - rr * s;
  }

  const double btper0 = btpar + ztemp1 * ctmp_re;
  const double btpar0 = sqrt(1.0 - btper0 / (tgam * tgam));

  k2pp += xks * (1.0 - 1.0 / btpar0) + xku;
  k2gg += ctmp_im / btpar0 / tgam - ez;
}


void genesis4_beamsolver_push_particles_gpu(int total_particles,
                                             const int* slice_id,
                                             double* x,
                                             double* y,
                                             const double* px,
                                             const double* py,
                                             double* gamma_arr,
                                             double* theta_arr,
                                             const double* ez_arr,
                                             const int* first,
                                             const int* ngrid,
                                             const int* fslice,
                                             const double* gridmax,
                                             const double* dgrid,
                                             const double* rtmp,
                                             const double* rharm,
                                             const double* const* field_re,
                                             const double* const* field_im,
                                             int nfield,
                                             double xks,
                                             double xku,
                                             double aw,
                                             double autophase,
                                             double delz,
                                             double u_ax,
                                             double u_ay,
                                             double u_kx,
                                             double u_ky,
                                             double u_gradx,
                                             double u_grady)
{
  // Keep the extended __device__ lambda in a namespace-scope helper.  NVCC
  // rejects extended device lambdas whose enclosing function is a private or
  // protected class member.
  amrex::ParallelFor(total_particles,
  [=] AMREX_GPU_DEVICE (int ip_global) noexcept {
    const int is = slice_id[ip_global];

    double gamma = gamma_arr[ip_global];
    double theta = theta_arr[ip_global] + autophase;

    const double xp = x[ip_global];
    const double yp = y[ip_global];
    const double pxp = px[ip_global];
    const double pyp = py[ip_global];

    const double dx = xp - u_ax;
    const double dy = yp - u_ay;
    const double awloc = 1.0 + 0.5 * (u_kx * dx * dx + u_ky * dy * dy)
                       + u_gradx * dx + u_grady * dy;

    const double btpar = 1.0 + pxp * pxp + pyp * pyp + aw * aw * awloc * awloc;
    const double ez = ez_arr[ip_global];

    double rpart_re_stack[kMaxBeamSolverFields];
    double rpart_im_stack[kMaxBeamSolverFields];
    for (int ifld = 0; ifld < kMaxBeamSolverFields; ++ifld) {
      rpart_re_stack[ifld] = 0.0;
      rpart_im_stack[ifld] = 0.0;
    }

    for (int ifld = 0; ifld < nfield; ++ifld) {
      const int ng = ngrid[ifld];
      const double gm = gridmax[ifld];
      if ((xp > -gm) && (xp < gm) && (yp > -gm) && (yp < gm)) {
        double wx = (xp + gm) / dgrid[ifld];
        double wy = (yp + gm) / dgrid[ifld];
        const int ix_raw = static_cast<int>(floor(wx));
        const int iy_raw = static_cast<int>(floor(wy));
        if (ix_raw < 0 || ix_raw >= ng - 1 || iy_raw < 0 || iy_raw >= ng - 1) {
          continue;
        }

        const int ix = ix_raw;
        const int iy = iy_raw;
        wx = 1.0 + static_cast<double>(ix) - wx;
        wy = 1.0 + static_cast<double>(iy) - wy;

        const int islice = (is + first[ifld]) % fslice[ifld];
        const int plane = ng * ng;
        int idx = ix + iy * ng;
        const int base = islice * plane;

        const double w00 = wx * wy;
        const double w10 = (1.0 - wx) * wy;
        const double w01 = wx * (1.0 - wy);
        const double w11 = (1.0 - wx) * (1.0 - wy);

        double cre = field_re[ifld][base + idx] * w00;
        double cim = field_im[ifld][base + idx] * w00;
        ++idx;
        cre += field_re[ifld][base + idx] * w10;
        cim += field_im[ifld][base + idx] * w10;
        idx += ng - 1;
        cre += field_re[ifld][base + idx] * w01;
        cim += field_im[ifld][base + idx] * w01;
        ++idx;
        cre += field_re[ifld][base + idx] * w11;
        cim += field_im[ifld][base + idx] * w11;

        const double scale = rtmp[ifld] * awloc;
        rpart_re_stack[ifld] = scale * cre;
        rpart_im_stack[ifld] = -scale * cim;
      }
    }

    double k2gg = 0.0;
    double k2pp = 0.0;
    double k3gg = 0.0;
    double k3pp = 0.0;

    genesis4_beamsolver_ode_gpu(gamma, theta, btpar, ez, xks, xku,
                                 nfield, rharm, rpart_re_stack, rpart_im_stack,
                                 k2gg, k2pp);

    double stpz = 0.5 * delz;
    gamma += stpz * k2gg;
    theta += stpz * k2pp;
    k3gg = k2gg;
    k3pp = k2pp;
    k2gg = 0.0;
    k2pp = 0.0;

    genesis4_beamsolver_ode_gpu(gamma, theta, btpar, ez, xks, xku,
                                 nfield, rharm, rpart_re_stack, rpart_im_stack,
                                 k2gg, k2pp);

    gamma += stpz * (k2gg - k3gg);
    theta += stpz * (k2pp - k3pp);
    k3gg /= 6.0;
    k3pp /= 6.0;
    k2gg *= -0.5;
    k2pp *= -0.5;

    genesis4_beamsolver_ode_gpu(gamma, theta, btpar, ez, xks, xku,
                                 nfield, rharm, rpart_re_stack, rpart_im_stack,
                                 k2gg, k2pp);

    stpz = delz;
    gamma += stpz * k2gg;
    theta += stpz * k2pp;
    k3gg -= k2gg;
    k3pp -= k2pp;
    k2gg *= 2.0;
    k2pp *= 2.0;

    genesis4_beamsolver_ode_gpu(gamma, theta, btpar, ez, xks, xku,
                                 nfield, rharm, rpart_re_stack, rpart_im_stack,
                                 k2gg, k2pp);

    gamma += stpz * (k3gg + k2gg / 6.0);
    theta += stpz * (k3pp + k2pp / 6.0);

    gamma_arr[ip_global] = gamma;
    theta_arr[ip_global] = theta;
  });

  amrex::Gpu::streamSynchronize();
}
} // namespace
#endif

BeamSolver::BeamSolver()
{
  onlyFundamental=false;
}

BeamSolver::~BeamSolver()= default;

void BeamSolver::advance(double delz, Beam *beam, vector<Field *> *field, Undulator *und) {
#ifdef GENESIS_USE_AMREX
  this->advance_gpu(delz, beam, field, und);
#else
  this->advance_cpu(delz, beam, field, und);
#endif
}

void BeamSolver::advance_cpu(double delz, Beam *beam, vector<Field *> *field, Undulator *und) {

  // here the harmonics needs to be taken into account

  vector<int> nfld;
  vector<double> rtmp;
  rpart.clear();
  rharm.clear();
  xks = 1;  // default value in the case that no field is defined

  for (int i = 0; i < static_cast<int>(field->size()); i++) {
    int harm = field->at(i)->getHarm();
    if ((harm == 1) || !onlyFundamental) {
      xks = field->at(i)->xks / static_cast<double>(harm); // fundamental field wavenumber used in ODE below
      nfld.push_back(i);
      rtmp.push_back(und->fc(harm) / field->at(i)->xks); // here the harmonics have to be taken care
      rpart.emplace_back(0);
      rharm.push_back(static_cast<double>(harm));
    }
  }
  xku = und->getku();
  if (xku == 0) { // in the case of drifts - the beam stays in phase if it has the reference energy
    // this requires that the phase slippage is not applied
    xku = xks * 0.5 / und->getGammaRef() / und->getGammaRef();
  }
  double aw = und->getaw();
  double autophase = und->autophase();

  // obtaining long range space charge field
  efield.longRange(beam, und->getGammaRef(), aw);  // defines the array beam->longESC

  // Runge Kutta solver to advance particle
  auto gammaz2 = und->getGammaRef()*und->getGammaRef()/(1+aw*aw);
  for (int is = 0; is < static_cast<int>(beam->beam.size()); is++) {
    // accumulate space charge field
    double eloss = -beam->longESC[is] / 511000; // convert eV to units of electron rest mass
    efield.shortRange(&beam->beam.at(is), beam->current.at(is), gammaz2, is);

    for (int ip = 0; ip < static_cast<int>(beam->beam.at(is).size()); ip++) {
      gamma = beam->beam.at(is).at(ip).gamma;
      theta = beam->beam.at(is).at(ip).theta + autophase; // add autophase here
      double x = beam->beam.at(is).at(ip).x;
      double y = beam->beam.at(is).at(ip).y;
      double px = beam->beam.at(is).at(ip).px;
      double py = beam->beam.at(is).at(ip).py;
      double awloc = und->faw(x, y); // get the transverse dependence of the undulator field
      btpar = 1 + px * px + py * py + aw * aw * awloc * awloc;
      ez = efield.getEField(ip) + eloss; // adding global long range space charge field to each particle
      cpart = 0;
      double wx, wy;
      int idx;

      for (int ifld = 0; ifld < static_cast<int>(nfld.size()); ifld++) {
        auto islice = (is + field->at(nfld[ifld])->first) % field->at(nfld[ifld])->field.size();
        if (field->at(nfld[ifld])->getLLGridpoint(x, y, &wx, &wy, &idx)) { // check whether particle is on grid
          cpart = field->at(nfld[ifld])->field[islice].at(idx) * wx * wy;
          idx++;
          cpart += field->at(nfld[ifld])->field[islice].at(idx) * (1 - wx) * wy;
          idx += field->at(nfld[ifld])->ngrid - 1;
          cpart += field->at(nfld[ifld])->field[islice].at(idx) * wx * (1 - wy);
          idx++;
          cpart += field->at(nfld[ifld])->field[islice].at(idx) * (1 - wx) * (1 - wy);
          rpart[ifld] = rtmp[ifld] * awloc * conj(cpart);
        } else {
          rpart[ifld] = 0;
        }
      }
      this->RungeKutta(delz);
      beam->beam.at(is).at(ip).gamma = gamma;
      beam->beam.at(is).at(ip).theta = theta;
    }
  }
}

#ifdef GENESIS_USE_AMREX
void BeamSolver::advance_gpu(double delz, Beam *beam, vector<Field *> *field, Undulator *und) {

  auto fallback_cpu = [&]() {
    cout << "BeamSolver::advance_gpu fallback to cpu"  << endl;
    if (beam != nullptr && beam->beamSoA != nullptr && beam->beamSoA->initialized) {
      beam->unpack_soa_to_beam();
    }
    this->advance_cpu(delz, beam, field, und);
    if (beam != nullptr && beam->beamSoA != nullptr && beam->beamSoA->initialized) {
      beam->pack_beam_to_soa();
    }
  };

  if (beam == nullptr || field == nullptr || und == nullptr) {
    return;
  }

  if (beam->beamSoA == nullptr || !beam->beamSoA->initialized) {
    beam->pack_beam_to_soa();
  }

  Genesis4BeamSoA* bsoa = beam->beamSoA;
  if (bsoa == nullptr || !bsoa->initialized || bsoa->total_particles <= 0 || bsoa->nslice <= 0) {
    fallback_cpu();
    return;
  }

  const int nslice = bsoa->nslice;
  const int total_particles = bsoa->total_particles;

  if (bsoa->slice_offsets.size() < static_cast<std::size_t>(nslice + 1) ||
      bsoa->slice_id.size() < static_cast<std::size_t>(total_particles) ||
      bsoa->x.size() < static_cast<std::size_t>(total_particles) ||
      bsoa->y.size() < static_cast<std::size_t>(total_particles) ||
      bsoa->px.size() < static_cast<std::size_t>(total_particles) ||
      bsoa->py.size() < static_cast<std::size_t>(total_particles) ||
      bsoa->gamma.size() < static_cast<std::size_t>(total_particles) ||
      bsoa->theta.size() < static_cast<std::size_t>(total_particles)) {
    fallback_cpu();
    return;
  }

  // Build the active harmonic/field list on host.  Only POD metadata and raw SoA
  // pointers are copied into GPU-visible vectors; no std::vector or class methods
  // are called inside device code.
  std::vector<int> h_first;
  std::vector<int> h_ngrid;
  std::vector<int> h_nslice;
  std::vector<double> h_gridmax;
  std::vector<double> h_dgrid;
  std::vector<double> h_rtmp;
  std::vector<double> h_rharm;
  std::vector<const double*> h_field_re;
  std::vector<const double*> h_field_im;

  double xks_local = 1.0;
  for (int i = 0; i < static_cast<int>(field->size()); ++i) {
    Field* fld = field->at(i);
    if (fld == nullptr) { continue; }

    const int harm = fld->getHarm();
    if ((harm != 1) && onlyFundamental) { continue; }

    if (fld->fieldSoA == nullptr || !fld->fieldSoA->initialized) {
      fld->pack_field_to_soa();
    }

    if (fld->fieldSoA == nullptr || !fld->fieldSoA->initialized ||
        fld->fieldSoA->field_re.size() == 0 || fld->fieldSoA->field_im.size() == 0 ||
        fld->field.size() == 0 || fld->ngrid <= 1) {
      fallback_cpu();
      return;
    }

    const int field_nslice = fld->fieldSoA->nslice > 0
                               ? fld->fieldSoA->nslice
                               : static_cast<int>(fld->field.size());
    const int plane = fld->ngrid * fld->ngrid;
    const std::size_t required_cells = static_cast<std::size_t>(field_nslice) *
                                       static_cast<std::size_t>(plane);
    if (fld->fieldSoA->field_re.size() < required_cells ||
        fld->fieldSoA->field_im.size() < required_cells) {
      fallback_cpu();
      return;
    }

    xks_local = fld->xks / static_cast<double>(harm);
    h_first.push_back(fld->first);
    h_ngrid.push_back(fld->ngrid);
    h_nslice.push_back(field_nslice);
    h_gridmax.push_back(fld->gridmax);
    h_dgrid.push_back(fld->dgrid);
    h_rtmp.push_back(und->fc(harm) / fld->xks);
    h_rharm.push_back(static_cast<double>(harm));
    h_field_re.push_back(fld->fieldSoA->field_re.data());
    h_field_im.push_back(fld->fieldSoA->field_im.data());
  }

  const int nfield = static_cast<int>(h_rharm.size());
  if (nfield > kMaxBeamSolverFields) {
    fallback_cpu();
    return;
  }

  double xku_local = und->getku();
  if (xku_local == 0.0) {
    const double gamma_ref = und->getGammaRef();
    xku_local = xks_local * 0.5 / gamma_ref / gamma_ref;
  }

  const double aw = und->getaw();
  const double autophase = und->autophase();
  const double gammaz2 = und->getGammaRef() * und->getGammaRef() / (1.0 + aw * aw);

  // The existing EFieldSolver still consumes Beam::beam.  Sync current SoA state
  // to AoS before evaluating long/short space charge.
  
  // beam->unpack_soa_to_beam();
  // efield.longRange(beam, und->getGammaRef(), aw);

  amrex::Gpu::HostVector<int> h_slice_offsets(nslice + 1);
  amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                        bsoa->slice_offsets.begin(),
                        bsoa->slice_offsets.begin() + nslice + 1,
                        h_slice_offsets.begin());
  amrex::Gpu::streamSynchronize();

  amrex::Gpu::HostVector<double> h_ez(total_particles, 0.0);
  for (int is = 0; is < nslice; ++is) {
    const int off0 = h_slice_offsets[is];
    const int off1 = h_slice_offsets[is + 1];
    const int np = off1 - off0;
    if (np <= 0) { continue; }

    const double eloss = -beam->longESC[is] / 511000.0;
    efield.shortRange(&beam->beam.at(is), beam->current.at(is), gammaz2, is);
    for (int ip = 0; ip < np; ++ip) {
      h_ez[off0 + ip] = efield.getEField(ip) + eloss;
    }
  }

  amrex::Gpu::DeviceVector<double> d_ez(total_particles);
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_ez.begin(), h_ez.end(), d_ez.begin());

  amrex::Gpu::HostVector<int> h_first_gpu(nfield);
  amrex::Gpu::HostVector<int> h_ngrid_gpu(nfield);
  amrex::Gpu::HostVector<int> h_nslice_gpu(nfield);
  amrex::Gpu::HostVector<double> h_gridmax_gpu(nfield);
  amrex::Gpu::HostVector<double> h_dgrid_gpu(nfield);
  amrex::Gpu::HostVector<double> h_rtmp_gpu(nfield);
  amrex::Gpu::HostVector<double> h_rharm_gpu(nfield);
  amrex::Gpu::HostVector<const double*> h_field_re_gpu(nfield);
  amrex::Gpu::HostVector<const double*> h_field_im_gpu(nfield);

  for (int i = 0; i < nfield; ++i) {
    h_first_gpu[i] = h_first[i];
    h_ngrid_gpu[i] = h_ngrid[i];
    h_nslice_gpu[i] = h_nslice[i];
    h_gridmax_gpu[i] = h_gridmax[i];
    h_dgrid_gpu[i] = h_dgrid[i];
    h_rtmp_gpu[i] = h_rtmp[i];
    h_rharm_gpu[i] = h_rharm[i];
    h_field_re_gpu[i] = h_field_re[i];
    h_field_im_gpu[i] = h_field_im[i];
  }

  amrex::Gpu::DeviceVector<int> d_first(nfield);
  amrex::Gpu::DeviceVector<int> d_ngrid(nfield);
  amrex::Gpu::DeviceVector<int> d_nslice(nfield);
  amrex::Gpu::DeviceVector<double> d_gridmax(nfield);
  amrex::Gpu::DeviceVector<double> d_dgrid(nfield);
  amrex::Gpu::DeviceVector<double> d_rtmp(nfield);
  amrex::Gpu::DeviceVector<double> d_rharm(nfield);
  amrex::Gpu::DeviceVector<const double*> d_field_re(nfield);
  amrex::Gpu::DeviceVector<const double*> d_field_im(nfield);

  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_first_gpu.begin(), h_first_gpu.end(), d_first.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_ngrid_gpu.begin(), h_ngrid_gpu.end(), d_ngrid.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_nslice_gpu.begin(), h_nslice_gpu.end(), d_nslice.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_gridmax_gpu.begin(), h_gridmax_gpu.end(), d_gridmax.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_dgrid_gpu.begin(), h_dgrid_gpu.end(), d_dgrid.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_rtmp_gpu.begin(), h_rtmp_gpu.end(), d_rtmp.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_rharm_gpu.begin(), h_rharm_gpu.end(), d_rharm.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_field_re_gpu.begin(), h_field_re_gpu.end(), d_field_re.begin());
  amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                        h_field_im_gpu.begin(), h_field_im_gpu.end(), d_field_im.begin());

  const int istep = und->getStep();
  const double u_ax = und->ax[istep];
  const double u_ay = und->ay[istep];
  const double u_kx = und->kx[istep];
  const double u_ky = und->ky[istep];
  const double u_gradx = und->gradx[istep];
  const double u_grady = und->grady[istep];

  genesis4_beamsolver_push_particles_gpu(total_particles,
                                          bsoa->slice_id.data(),
                                          bsoa->x.data(),
                                          bsoa->y.data(),
                                          bsoa->px.data(),
                                          bsoa->py.data(),
                                          bsoa->gamma.data(),
                                          bsoa->theta.data(),
                                          d_ez.data(),
                                          d_first.data(),
                                          d_ngrid.data(),
                                          d_nslice.data(),
                                          d_gridmax.data(),
                                          d_dgrid.data(),
                                          d_rtmp.data(),
                                          d_rharm.data(),
                                          d_field_re.data(),
                                          d_field_im.data(),
                                          nfield,
                                          xks_local,
                                          xku_local,
                                          aw,
                                          autophase,
                                          delz,
                                          u_ax,
                                          u_ay,
                                          u_kx,
                                          u_ky,
                                          u_gradx,
                                          u_grady);

}
#endif

void BeamSolver::RungeKutta(double delz) {
  // Runge Kutta Solver 4th order - taken from pushp from the old Fortran source
  // first step
  k2gg = 0;
  k2pp = 0;
  this->ODE(gamma, theta);

  // second step
  double stpz = 0.5 * delz;
  gamma += stpz * k2gg;
  theta += stpz * k2pp;
  k3gg = k2gg;
  k3pp = k2pp;
  k2gg = 0;
  k2pp = 0;
  this->ODE(gamma, theta);

  // third step
  gamma += stpz * (k2gg - k3gg);
  theta += stpz * (k2pp - k3pp);
  k3gg /= 6;
  k3pp /= 6;
  k2gg *= -0.5;
  k2pp *= -0.5;
  this->ODE(gamma, theta);

  // fourth step
  stpz = delz;
  gamma += stpz * k2gg;
  theta += stpz * k2pp;
  k3gg -= k2gg;
  k3pp -= k2pp;
  k2gg *= 2;
  k2pp *= 2;
  this->ODE(gamma, theta);
  gamma += stpz * (k3gg + k2gg / 6.0);
  theta += stpz * (k3pp + k2pp / 6.0);
}

void BeamSolver::ODE(double tgam,double tthet) {
  // differential equation for longitudinal motion
  double ztemp1 = -2. / xks;
  complex<double> ctmp = 0;

  for (int i = 0; i < static_cast<int>(rpart.size()); i++) {
    ctmp += rpart[i] * complex<double>(cos(rharm[i] * tthet), -sin(rharm[i] * tthet));
  }
  double btper0 = btpar + ztemp1 * ctmp.real(); //perpendicular velocity
  double btpar0 = sqrt(1. - btper0 / (tgam * tgam)); //parallel velocity

#ifdef G4_DBGDIAG
  // CL: detect negative radicands as NaN theta values can be the result
  double btpar0_sq=1.-btper0/(tgam*tgam); //(parallel velocity)^2
  if(btpar0_sq<0) {
    cout << "DBGDIAG(BeamSolver::ODE): error, negative radicand detected" << endl;
  }
#endif

  k2pp += xks * (1. - 1. / btpar0) + xku;     //dtheta/dz
  k2gg += ctmp.imag() / btpar0 / tgam - ez;   //dgamma/dz
}

void BeamSolver::checkAllocation(unsigned long nslice) {
  efield.allocateForOutput(nslice);
}
