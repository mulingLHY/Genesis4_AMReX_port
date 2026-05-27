#ifndef GENESIS_DIAGNOSTICCUDA_H
#define GENESIS_DIAGNOSTICCUDA_H

#ifdef GENESIS_USE_AMREX

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Diagnostic.h"

class Beam;
class Field;
struct DiagnosticAMReXScratch;

class DiagnosticCUDA {
 public:
  DiagnosticCUDA();
  ~DiagnosticCUDA();

  void init(FilterDiagnostics &);
  bool calc(Beam *, std::vector<Field *> *, double, Diagnostic &);

 private:
  bool calcBeam(Beam *, std::map<std::string, std::vector<double> > &);
  bool calcField(Field *, std::map<std::string, std::vector<double> > &);

  FilterDiagnostics filter_;
  int iz_;
  std::unique_ptr<DiagnosticAMReXScratch> scratch_;
};

#endif

#endif
