#include <climits>
#include <cmath>
#include <complex>
#include <iostream>
#include <sstream>

#include <mpi.h>

#include "Control.h"
#include "writeFieldHDF5.h"
#include "writeBeamHDF5.h"

#ifdef GENESIS_USE_AMREX
#include "Genesis4FieldSoA.h"
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#endif

#ifdef GENESIS_USE_AMREX
namespace {

void g4_pack_field_slice_to_interleaved_gpu(const double* field_re,
                                            const double* field_im,
                                            double* buffer,
                                            int slice,
                                            int ncells)
{
    const int base = slice * ncells;
    amrex::ParallelFor(ncells,
        [=] AMREX_GPU_DEVICE (int i) noexcept {
            buffer[2*i    ] = field_re[base + i];
            buffer[2*i + 1] = field_im[base + i];
        });
}

void g4_unpack_interleaved_to_field_slice_gpu(double* field_re,
                                              double* field_im,
                                              const double* buffer,
                                              int slice,
                                              int ncells)
{
    const int base = slice * ncells;
    amrex::ParallelFor(ncells,
        [=] AMREX_GPU_DEVICE (int i) noexcept {
            field_re[base + i] = buffer[2*i    ];
            field_im[base + i] = buffer[2*i + 1];
        });
}

void g4_zero_field_slice_gpu(double* field_re,
                             double* field_im,
                             int slice,
                             int ncells)
{
    const int base = slice * ncells;
    amrex::ParallelFor(ncells,
        [=] AMREX_GPU_DEVICE (int i) noexcept {
            field_re[base + i] = 0.0;
            field_im[base + i] = 0.0;
        });
}

} // namespace
#endif

Control::Control()
{
    nwork=0;
    work=nullptr;
#ifdef GENESIS_USE_AMREX
    gpu_nwork=0;
    gpu_work=nullptr;
    gpu_pinned_nwork=0;
    gpu_pinned_work=nullptr;
#endif
}

Control::~Control()
{
#ifdef GENESIS_USE_AMREX
    if (gpu_work != nullptr) {
        amrex::The_Arena()->free(gpu_work);
        gpu_work = nullptr;
        gpu_nwork = 0;
    }
    if (gpu_pinned_work != nullptr) {
        amrex::The_Pinned_Arena()->free(gpu_pinned_work);
        gpu_pinned_work = nullptr;
        gpu_pinned_nwork = 0;
    }
#endif
    delete[] work;
}

bool Control::applyMarker(Beam *beam, vector<Field *> *field, Undulator *und, bool& error_IO)
{
    error_IO = false;
    /* error_IO==true signals error during a requested dump */
    bool sort=false;
    int marker=und->getMarker();

    // possible file names contain number of current integration step
    stringstream sroot;
    string basename;
    int istepz=und->getStep();
    sroot << "." << istepz;
    basename=root+sroot.str();

    if ((marker & 1) != 0){
        WriteFieldHDF5 dump;
        if(dump.write(basename,field)) {
            /* register field dump => it will be reported in list of dumps generated during current "&track" command */
            string fn;
            fn = basename + ".fld.h5";
            /* file extension as added in WriteFieldHDF5::write (TODO: need to implement proper handling of harmonic field dumping) */
            und->fielddumps_filename.push_back(fn);
            und->fielddumps_intstep.push_back(istepz);
        } else {
            /* IO error: do not add filename to list */
            error_IO = true;
            if(rank==0) {
                cout << " write operation was not successful!" << endl;
            }
        }
    }
    if ((marker & 2) != 0){
        WriteBeamHDF5 dump;
        if(dump.write(basename,beam,1)) { // use stride of 1 -> all particles are dump
            /* register beam dump => it will be reported in list of dumps generated during current "&track" command */
            string fn;
            fn = basename + ".par.h5";
            /* file extension as added in WriteBeamHDF5::write */
            und->beamdumps_filename.push_back(fn);
            und->beamdumps_intstep.push_back(istepz);
        } else {
            /* IO error: do not add filename to list */
            error_IO = true;
            if(rank==0) {
                cout << " write operation was not successful!" << endl;
            }
        }
    }
    if ((marker & 4) != 0){
        sort=true; // sorting is deferred after the particles have been pushed by Runge-Kutta
    }
    // bit value 8 is checked in und->advance()
    return sort;
}

#if 0
// .out.h5 file is now written in class Diagnostic
void Control::output(Beam *beam, vector<Field *> *field, Undulator *und, Diagnostic &diag)
{
    Output *out=new Output;
    string file=root.append(".out.h5");
    out->open(file,noffset,nslice);
    out->writeGlobal(und,und->getGammaRef(),reflen,sample,slen,one4one,timerun,scanrun,ntotal);
    out->writeLattice(beam,und);
    for (unsigned int i=0; i<field->size();i++){
        out->writeFieldBuffer(field->at(i));
    }
    out->writeBeamBuffer(beam);
    out->close();
    delete out;
    return;
}
#endif

bool Control::init(int inrank, int insize, const string in_rootname,
                   Beam *beam, vector<Field *> *field, Undulator *und,
                   bool inTime, bool inScan, bool inPeriodic)
{
    rank=inrank;
    size=insize;
    periodic = inPeriodic;
    root = in_rootname;
    one4one=beam->one4one;
    reflen=beam->reflength;
    sample=beam->slicelength/reflen;
    timerun=inTime;
    scanrun=inScan;

    // cross check simulation size
    nslice=beam->beam.size();
    noffset=rank*nslice;
    ntotal=size*nslice; // all cores have the same amount of slices
    slen=ntotal*sample*reflen;
    if (rank==0){
        if(scanrun) {
            cout << "Scan run with " << ntotal << " slices" << endl;
        } else {
            if(timerun) {
                cout << "Time-dependent run with " << ntotal << " slices"
                     << " for a time window of " << slen*1e6 << " microns" << endl;
                if (periodic) {
                    cout << "Periodic boundary condition of time window enabled" << endl;
                }
            } else {
                cout << "Steady-state run" << endl;
            }
        }
    }
    for (auto & fld : *field){
        fld->resetSlippage();
    }
    beam->checkBeforeTracking();
    return true;
}

void Control::applySlippage(double slippage, Field *field)
{
#ifdef GENESIS_USE_AMREX
    if (field != nullptr && field->fieldSoA != nullptr && field->fieldSoA->initialized) {
        applySlippage_gpu(slippage, field);
        return;
    }
#endif
    applySlippage_cpu(slippage, field);
}

void Control::applySlippage_cpu(double slippage, Field *field)
{
    if (timerun==false) { return; }

    // update accumulated slippage
    field->accuslip+=slippage;

    // number of grid points in field supplied by caller
    long long ncells = static_cast<long long>(field->ngrid) * static_cast<long long>(field->ngrid);

    // if needed, allocate working space for MPI data transfer
    // NOTE: the size of the buffer is determined by the largest field seen so far
    // (relevant when there are multiple fields of different number of grid points)
    if(nwork < 2*ncells){
        delete[] work;
        nwork = 2*ncells; // 1 complex number <=> 2 doubles
        work=new double [nwork];
    }

    // following routine is applied if the required slippage is larger than 80% of the sampling size
    int direction=1;
    while(std::abs(field->accuslip)>(sample*0.8)){
        // check for abnormal direction of slippage (backwards slippage)
        if (field->accuslip<0) {direction=-1;}
        field->accuslip-=sample*direction;

        // get adjacent node before and after in chain
        int rank_next=rank+1;
        int rank_prev=rank-1;
        if (rank_next >= size ) { rank_next=0; }
        if (rank_prev < 0 ) { rank_prev = size-1; }

        // for inverse direction swap targets
        if (direction<0) {
            int tmp=rank_next;
            rank_next=rank_prev;
            rank_prev=tmp;
        }

        int tag=1;

        // get slice which is transmitted
        auto last=(field->first+field->field.size()-1) % field->field.size();
        // get first slice for inverse direction
        if (direction<0){
            last=(last+1) % field->field.size(); // this actually first because it is sent backwards
        }

        // Prevent transfer sizes resulting in overflow (MPI_send argument 'count' has data type 'int').
        // For typical transverse grid sizes, this is not a relevant limitation.
        // (All MPI processes have identical transverse field parameters.)
        if(2*ncells > INT_MAX) {
            if(rank==0) {
                cout << "Large field mesh size results in request for MPI transfer size exceeding INT_MAX, exiting." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        MPI_Status status;
        if (size>1) {
            if ( (rank % 2)==0 ){
                // even nodes are sending first and then receiving field
                for (int i=0; i<ncells; i++){
                    work[2*i]   =field->field[last].at(i).real();
                    work[2*i+1] =field->field[last].at(i).imag();
                }
                MPI_Send(work,2*ncells, /* <= number of DOUBLES */ MPI_DOUBLE,rank_next,tag,MPI_COMM_WORLD);
                MPI_Recv(work,2*ncells, MPI_DOUBLE,rank_prev,tag,MPI_COMM_WORLD,&status);
                for (int i=0; i<ncells; i++){
                    complex<double> ctemp=complex<double> (work[2*i],work[2*i+1]);
                    field->field[last].at(i)=ctemp;
                }
            } else {
                // odd nodes are receiving first and then sending
                MPI_Recv(work,2*ncells, /* <= number of DOUBLES */ MPI_DOUBLE,rank_prev,tag,MPI_COMM_WORLD,&status);
                for (int i=0; i<ncells; i++){
                    complex<double> ctemp=complex<double> (work[2*i],work[2*i+1]);
                    work[2*i]   =field->field[last].at(i).real();
                    work[2*i+1] =field->field[last].at(i).imag();
                    field->field[last].at(i)=ctemp;
                }
                MPI_Send(work,2*ncells,MPI_DOUBLE,rank_next,tag,MPI_COMM_WORLD);
            }
        }

        // first node has empty field slipped into the time window
        if (!periodic) {
            if ((rank==0) && (direction >0)){
                for (int i=0; i<ncells; i++){
                    field->field[last].at(i)=complex<double> (0,0);
                }
            }
            if ((rank==(size-1)) && (direction <0)){
                for (int i=0; i<ncells; i++){
                    field->field[last].at(i)=complex<double> (0,0);
                }
            }
        }

        // last was the last slice to be transmitted to the succeeding node and then filled with the
        // field from the preceding node, making it now the start of the field record.
        field->first=last;
        if (direction<0){
            field->first=(last+1) % field->field.size();
        }
    }
}

#ifdef GENESIS_USE_AMREX
void Control::applySlippage_gpu(double slippage, Field *field)
{
    if (timerun==false || field == nullptr || field->fieldSoA == nullptr || !field->fieldSoA->initialized) {
        return;
    }

    // Update scalar slippage on host.  No field data is touched until the same
    // threshold condition used by the original CPU routine is crossed.
    field->accuslip += slippage;

    Genesis4FieldSoA* soa = field->fieldSoA;
    const int nslices = soa->nslice;
    const int ngrid = soa->ngrid;
    if (nslices <= 0 || ngrid <= 0) {
        return;
    }

    const long long ncells_ll = static_cast<long long>(ngrid) * static_cast<long long>(ngrid);
    if(2*ncells_ll > INT_MAX) {
        if(rank==0) {
            cout << "Large field mesh size results in request for MPI transfer size exceeding INT_MAX, exiting." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    const int ncells = static_cast<int>(ncells_ll);

    // Two interleaved device buffers and two pinned host buffers are enough to preserve
    // the original odd/even MPI exchange semantics:
    //   sendbuf = local outgoing plane, recvbuf = incoming plane from neighbor.
    // MPI is intentionally issued on pinned host buffers instead of managed/device memory.
    const long long needed = 4LL * static_cast<long long>(ncells);
    if (gpu_nwork < needed) {
        if (gpu_work != nullptr) {
            amrex::The_Arena()->free(gpu_work);
            gpu_work = nullptr;
        }
        gpu_work = static_cast<double*>(amrex::The_Arena()->alloc(sizeof(double) * static_cast<std::size_t>(needed)));
        gpu_nwork = needed;
    }
    if (gpu_pinned_nwork < needed) {
        if (gpu_pinned_work != nullptr) {
            amrex::The_Pinned_Arena()->free(gpu_pinned_work);
            gpu_pinned_work = nullptr;
        }
        gpu_pinned_work = static_cast<double*>(amrex::The_Pinned_Arena()->alloc(sizeof(double) * static_cast<std::size_t>(needed)));
        gpu_pinned_nwork = needed;
    }

    double* d_sendbuf = gpu_work;
    double* d_recvbuf = gpu_work + 2LL * static_cast<long long>(ncells);
    double* h_sendbuf = gpu_pinned_work;
    double* h_recvbuf = gpu_pinned_work + 2LL * static_cast<long long>(ncells);
    double* field_re = soa->field_re.data();
    double* field_im = soa->field_im.data();

    int direction=1;
    while(std::abs(field->accuslip)>(sample*0.8)){
        if (field->accuslip<0) { direction=-1; }
        else { direction=1; }
        field->accuslip-=sample*direction;

        int rank_next=rank+1;
        int rank_prev=rank-1;
        if (rank_next >= size ) { rank_next=0; }
        if (rank_prev < 0 ) { rank_prev = size-1; }

        if (direction<0) {
            int tmp=rank_next;
            rank_next=rank_prev;
            rank_prev=tmp;
        }

        int tag=1;

        int last=(field->first+nslices-1) % nslices;
        if (direction<0){
            last=(last+1) % nslices;
        }

        MPI_Status status;
        if (size>1) {
            // Pack one plane only.  This is intentionally much cheaper than unpacking/packing
            // the whole Field::field array in Gencore every step.
            g4_pack_field_slice_to_interleaved_gpu(field_re, field_im, d_sendbuf, last, ncells);
            amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                                  d_sendbuf, d_sendbuf + 2*ncells,
                                  h_sendbuf);
            amrex::Gpu::streamSynchronize();

            if ((rank % 2)==0) {
                MPI_Send(h_sendbuf, 2*ncells, MPI_DOUBLE, rank_next, tag, MPI_COMM_WORLD);
                MPI_Recv(h_recvbuf, 2*ncells, MPI_DOUBLE, rank_prev, tag, MPI_COMM_WORLD, &status);
                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_recvbuf, h_recvbuf + 2*ncells,
                                      d_recvbuf);
                amrex::Gpu::streamSynchronize();
                g4_unpack_interleaved_to_field_slice_gpu(field_re, field_im, d_recvbuf, last, ncells);
            } else {
                MPI_Recv(h_recvbuf, 2*ncells, MPI_DOUBLE, rank_prev, tag, MPI_COMM_WORLD, &status);
                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                      h_recvbuf, h_recvbuf + 2*ncells,
                                      d_recvbuf);
                amrex::Gpu::streamSynchronize();
                g4_unpack_interleaved_to_field_slice_gpu(field_re, field_im, d_recvbuf, last, ncells);
                amrex::Gpu::streamSynchronize();
                MPI_Send(h_sendbuf, 2*ncells, MPI_DOUBLE, rank_next, tag, MPI_COMM_WORLD);
            }
        }

        if (!periodic) {
            if ((rank==0) && (direction >0)){
                g4_zero_field_slice_gpu(field_re, field_im, last, ncells);
            }
            if ((rank==(size-1)) && (direction <0)){
                g4_zero_field_slice_gpu(field_re, field_im, last, ncells);
            }
        }

        field->first=last;
        if (direction<0){
            field->first=(last+1) % nslices;
        }
    }
}
#endif
