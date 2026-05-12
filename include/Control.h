#ifndef __GENESIS_CONTROL__
#define __GENESIS_CONTROL__

#include <string>
#include <vector>

#include "Field.h"
#include "Beam.h"
#include "Undulator.h"
#include "HDF5base.h"
#include "Output.h"
#include "Diagnostic.h"

using namespace std;

class Control : public HDF5Base{
 public:
   Control();
   virtual ~Control();
   void applySlippage(double, Field *);
   bool applyMarker(Beam *, vector<Field *> *, Undulator *, bool&);
   bool init(int, int, const std::string, Beam *, vector<Field *> *, Undulator *, bool, bool, bool);
   // void output(Beam *, vector<Field *> *, Undulator *, Diagnostic &);

 private:
   void applySlippage_cpu(double, Field *);
#ifdef GENESIS_USE_AMREX
   void applySlippage_gpu(double, Field *);
#endif

   bool timerun,scanrun,one4one,periodic;
   int nslice,ntotal,noffset;
   int rank, size;
   double sample,reflen,slen;
   int nzout;
   int nwork;
   double *work;
#ifdef GENESIS_USE_AMREX
   long long gpu_nwork;
   double *gpu_work;
   long long gpu_pinned_nwork;
   double *gpu_pinned_work;
#endif
   string root;
};

#endif
