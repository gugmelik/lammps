/***************************************************************************
                                 lj_ext.cpp
                             -------------------
                            W. Michael Brown (ORNL)

  Functions for LAMMPS access to lj/cut acceleration routines.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : brownw@ornl.gov
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_mlip.h"

using namespace std;
using namespace LAMMPS_AL;

static MLMTP<PRECISION,ACC_PRECISION> MLMTPF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int mlip_gpu_init(const int ntypes, double **cutsq, int host_alpha_moments_count,
                 int host_alpha_index_basic_count, int host_alpha_times_count,
                 int (*host_alpha_index_basic)[4], int (*host_alpha_index_times_count)[4], int *host_alpha_moment_mapping, const int inum,
                 const int nall, const int max_nbors,  const int maxspecial,
                 const double cell_size, int &gpu_mode, FILE *screen) {
  MLMTPF.clear();
  gpu_mode=MLMTPF.device->gpu_mode();
  double gpu_split=MLMTPF.device->particle_split();
  int first_gpu=MLMTPF.device->first_device();
  int last_gpu=MLMTPF.device->last_device();
  int world_me=MLMTPF.device->world_me();
  int gpu_rank=MLMTPF.device->gpu_rank();
  int procs_per_gpu=MLMTPF.device->procs_per_gpu();

  MLMTPF.device->init_message(screen,"mlip",first_gpu,last_gpu);

  bool message=false;
  if (MLMTPF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=MLMTPF.init(ntypes, cutsq, host_alpha_moments_count,
                       host_alpha_index_basic_count, host_alpha_times_count,
                       host_alpha_index_basic, host_alpha_index_times_count, host_alpha_moment_mapping, inum, nall, 300,
                       maxspecial, cell_size, gpu_split, screen);

  MLMTPF.device->world_barrier();
  if (message)
    fprintf(screen,"Done.\n");

  for (int i=0; i<procs_per_gpu; i++) {
    if (message) {
      if (last_gpu-first_gpu==0)
        fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
      else
        fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                last_gpu,i);
      fflush(screen);
    }
    if (gpu_rank==i && world_me!=0)
      init_ok=MLMTPF.init(ntypes, cutsq, host_alpha_moments_count,
                        host_alpha_index_basic_count, host_alpha_times_count,
                        host_alpha_index_basic, host_alpha_index_times_count, host_alpha_moment_mapping, inum, nall, 300, maxspecial,
                        cell_size, gpu_split, screen);

    MLMTPF.device->gpu_barrier();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    MLMTPF.estimate_gpu_overhead();
  return init_ok;
}

// ---------------------------------------------------------------------------
// Copy updated coeffs from host to device
// ---------------------------------------------------------------------------
/*void ljl_gpu_reinit(const int ntypes, double **cutsq, double **host_lj1,
                    double **host_lj2, double **host_lj3, double **host_lj4,
                    double **offset) {
  int world_me=MLMTPF.device->world_me();
  int gpu_rank=MLMTPF.device->gpu_rank();
  int procs_per_gpu=MLMTPF.device->procs_per_gpu();

  if (world_me==0)
    MLMTPF.reinit(ntypes, cutsq, host_lj1, host_lj2, host_lj3, host_lj4, offset);
  MLMTPF.device->world_barrier();

  for (int i=0; i<procs_per_gpu; i++) {
    if (gpu_rank==i && world_me!=0)
      MLMTPF.reinit(ntypes, cutsq, host_lj1, host_lj2, host_lj3, host_lj4, offset);
    MLMTPF.device->gpu_barrier();
  }
}*/

void ljl_gpu_clear() {
  MLMTPF.clear();
}

int ** mlip_gpu_compute_n(const int ago, const int inum_full,
                        const int nall, double **host_x, int *host_type,
                        double *sublo, double *subhi, tagint *tag, int **nspecial,
                        tagint **special, const bool eflag, const bool vflag,
                        const bool eatom, const bool vatom, int &host_start,
                        int **ilist, int **jnum, const double cpu_time,
                        bool &success) {
  return MLMTPF.compute(ago, inum_full, nall, host_x, host_type, sublo,
                       subhi, tag, nspecial, special, eflag, vflag, eatom,
                       vatom, host_start, ilist, jnum, cpu_time, success);
}

void mlip_gpu_compute(const int ago, const int inum_full, const int nall,
                     double **host_x, int *host_type, int *ilist, int *numj,
                     int **firstneigh, const bool eflag, const bool vflag,
                     const bool eatom, const bool vatom, int &host_start,
                     const double cpu_time, bool &success) {
  MLMTPF.compute(ago,inum_full,nall,host_x,host_type,ilist,numj,
                firstneigh,eflag,vflag,eatom,vatom,host_start,cpu_time,success);
}

double mlip_gpu_bytes() {
  return MLMTPF.host_memory_usage();
}


