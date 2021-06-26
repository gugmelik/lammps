/* ----------------------------------------------------------------------
 *   This is the MLIP-LAMMPS interface
 *   MLIP is a software for Machine Learning Interatomic Potentials
 *   distributed by A. Shapeev, Skoltech (Moscow)
 *   Contributors: Evgeny Podryabinkin

   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   LAMMPS is distributed under a GNU General Public License
   and is not a part of MLIP.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Evgeny Podryabinkin (Skoltech)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mlip,PairMLIP)

#else

#ifndef LMP_PAIR_MLIP_H
#define LMP_PAIR_MLIP_H

#include <stdio.h>
#include "pair.h"
#include "multidimensional_arrays.h"
#include "radial_basis.h"
#include "vector3.h"

//extern void MLIP_init(const char*, const char*, int, double&, int&);
//extern void MLIP_calc_cfg(int, double*, double**, int*, int*, double&, double**, double*);
//extern void MLIP_calc_nbh(int, int*, int*, int**, int, int, double**, int*, double**, double&, double*, double**);
//extern void MLIP_finalize();

namespace LAMMPS_NS {

struct Neighborhood
{
	int count;                      //  number of neighbors
	std::vector<int> inds;          //	array of indices of neighbor atoms
	std::vector<Vector3> vecs;     //	array of relative positions of the neighbor atoms
	std::vector<double> dists;      //	array of distances to the neighbor atoms	
	std::vector<int> types;
	int my_type;
};

class PairMLIP : public Pair {
 public:
  int alpha_count;								//!< Basis functions count 
  int alpha_scalar_moments;						//!< = alpha_count-1 (MTP-basis except constant function)
  int radial_func_count;							//!< number of radial basis functions used
  int species_count;							//!< number of components present in the potential
  double cutoff;

  PairMLIP(class LAMMPS *);
  virtual ~PairMLIP();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);

  void CalcSiteEnergyDers(const Neighborhood& nbh);
  void Load(const std::string& filename);

  std::vector<double> LinCoeff();								//returns linear coefficients
  
  void DistributeCoeffs();									//Combine radial and linear coefficients in one arraya


  double scaling = 1.0; //!< how to scale moments

	std::vector<double> regression_coeffs;		
	std::vector<double> linear_coeffs;

	std::vector<double> linear_mults;					//!< array of multiplers for basis functions

	AnyRadialBasis* p_RadialBasis = nullptr;		//!< pointer to RadialBasis

	double buff_site_energy_;						//!< Temporal variable storing site energy after its calculation for a certain neighborhood
	std::vector<Vector3> buff_site_energy_ders_;    //!< Temporal variable storing derivatives of site energy w.r.t. positions of neghboring atoms after their calculation for a certain neighborhood

	

 protected:
  //int mode; // 0 - nbh mode (can't learn on the fly), 1 - cfg mode (typically for non-parallel lammps)
  char MLIPsettings_filename[1000];
  //char MLIPlog_filename[1000];
  double cutoffsq;
  void allocate();

  void MemAlloc();

  void ReadLinearCoeffs(std::ifstream& ifs);				//Read linear regression coeffs from MTP file
  void WriteLinearCoeffs(std::ofstream& ofs);				//Write linear regression coeffs to MTP file


	int alpha_moments_count;					//	/=================================================================================================
	int alpha_index_basic_count;				//	|                                                                                                |
	int(*alpha_index_basic)[4];					//	|   Internal representation of Moment Tensor Potential basis                                     |
	int alpha_index_times_count;				//	|   These items is required for calculation of basis functions values and their derivatives      |
	int(*alpha_index_times)[4];					//	|                                                                                                |
	int *alpha_moment_mapping;					//	\=================================================================================================
	std::string pot_desc;
	std::string rbasis_type;

	double *moment_vals; //!< Array of basis function values calculated for certain atomic neighborhood

	Array3D moment_jacobian_;
	std::vector<double> site_energy_ders_wrt_moments_;
	std::vector<double> dist_powers_;
	std::vector<Vector3> coords_powers_;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

*/
