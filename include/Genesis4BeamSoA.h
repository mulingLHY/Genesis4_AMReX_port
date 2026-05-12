#ifndef GENESIS4BEAMSOA_H_
#define GENESIS4BEAMSOA_H_

#ifdef GENESIS_USE_AMREX
#include <AMReX_GpuContainers.H>

struct Genesis4BeamSoA{
    amrex::Gpu::DeviceVector<double> x;
    amrex::Gpu::DeviceVector<double> y;
    amrex::Gpu::DeviceVector<double> px;
    amrex::Gpu::DeviceVector<double> py;
    amrex::Gpu::DeviceVector<double> gamma;
    amrex::Gpu::DeviceVector<double> theta;

    amrex::Gpu::DeviceVector<int> slice_id;
    amrex::Gpu::DeviceVector<int> slice_offsets;

    int nslice = 0;
    int total_particles = 0;
    bool initialized = false;
};
#endif // GENESIS_USE_AMREX

#endif // GENESIS4BEAMSOA_H_
