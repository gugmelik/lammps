/***************************************************************************
                                    lj_smooth.h
                             -------------------
                            Gurgen Melikyan (HSE University)
  Class for acceleration of the lj/smooth pair style.
 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________
    begin                :
    email                : gkmelikyan@edu.hse.ru
 ***************************************************************************/

#ifndef LAL_LJ_SMOOTH_H
#define LAL_LJ_SMOOTH_H

#include "lal_base_atomic.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class MLMTP : public BaseAtomic<numtyp, acctyp> {
 public:
  MLMTP();
  ~MLMTP();

  /// Clear any previous data and set up for a new LAMMPS run
  /** \param max_nbors initial number of rows in the neighbor matrix
    * \param cell_size cutoff + skin
    * \param gpu_split fraction of particles handled by device
    *
    * Returns:
    * -  0 if successful
    * - -1 if fix gpu not found
    * - -3 if there is an out of memory error
    * - -4 if the GPU library was not compiled for GPU
    * - -5 Double precision is not supported on card **/
  int init(const int ntypes, double **host_cutsq, int host_alpha_moments_count,
           int host_alpha_index_basic_count, int host_alpha_times_count,
           int (*host_alpha_index_basic)[4], int (*host_alpha_index_times_count)[4], int *host_alpha_moment_mapping,
           const int nlocal, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           const double gpu_split, FILE *screen);

  /// Send updated coeffs from host to device (to be compatible with fix adapt)
  /*void reinit(const int ntypes, double **host_cutsq,
              double **host_lj1, double **host_lj2, double **host_lj3,
              double **host_lj4, double **host_offset,
              double **host_ljsw0, double **host_ljsw1, double **host_ljsw2, 
              double **host_ljsw3, double **host_ljsw4,
              double **cut_inner, double **cut_inner_sq);*/

  /// Clear all host and device data
  /** \note This is called at the beginning of the init() routine **/
  void clear();

  /// Returns memory usage on device per atom
  int bytes_per_atom(const int max_nbors) const;

  /// Total host memory used by library for pair style
  double host_memory_usage() const;

  // --------------------------- TYPE DATA --------------------------

  /// alpha_coeff.x = alpha_moments_count, alpha_coeff.y = alpha_index_basic_count, alpha_coeff.z = alpha_index_times_count
  UCL_D_Vec<numtyp4> alpha_coeff;
  

  /// If atom type constants fit in shared memory, use fast kernels
  bool shared_types;

  /// Number of atom types
  int _lj_types;

 private:
  bool _allocated;
  int loop(const int _eflag, const int _vflag);
};

}

#endif