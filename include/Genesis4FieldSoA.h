#ifndef GENESIS4FIELDSOA_H_
#define GENESIS4FIELDSOA_H_

#ifdef GENESIS_USE_AMREX
#include <AMReX_GpuContainers.H>

struct Genesis4FieldSoA {
    // Flattened field data: real and imaginary parts separated
    // Total cells = nslice * ngrid * ngrid
    amrex::Gpu::DeviceVector<double> field_re;
    amrex::Gpu::DeviceVector<double> field_im;

    int nslice = 0;
    int ngrid = 0;
    int total_cells = 0;
    bool initialized = false;
};
#endif // GENESIS_USE_AMREX

#endif // GENESIS4FIELDSOA_H_