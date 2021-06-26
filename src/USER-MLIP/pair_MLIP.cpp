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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include "pair_MLIP.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace std;


#define MAXLINE 1024


/* ---------------------------------------------------------------------- */

PairMLIP::PairMLIP(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;

  single_enable = 0;

  allocated = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMLIP::~PairMLIP()
{
  if (copymode) return;

  if (allocated) {
      memory->destroy(setflag);
      memory->destroy(cutsq);
  }
  
  if (moment_vals != NULL) delete[] moment_vals;

  if (alpha_moment_mapping != NULL) delete[] alpha_moment_mapping;
  if (alpha_index_times != NULL) delete[] alpha_index_times;
  if (alpha_index_basic != NULL) delete[] alpha_index_basic;

  moment_vals = NULL;
  alpha_moment_mapping = NULL;
  alpha_index_times = NULL;
  alpha_index_basic = NULL;

  
  if (p_RadialBasis!=NULL)
  delete p_RadialBasis;
}

/* ---------------------------------------------------------------------- */

void PairMLIP::compute(int eflag, int vflag)
{
	int i,j,ii,jj,inum,jnum,itype,jtype,itable;
	double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
	double r;
	int *ilist,*jlist,*numneigh,**firstneigh;
	Neighborhood nbh;
	double energy = 0;
	double *p_site_en = NULL;
	double* p_site_energy_ders;
	double **p_site_virial = NULL;

	double **x = atom->x;
	double **f = atom->f;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int newton_pair = force->newton_pair;

	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	if (eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

	// nbh version

	if (eflag_atom) p_site_en = &eatom[0];
	if (vflag_atom) p_site_virial = vatom;
  

	for (ii = 0; ii < inum; ii++) 
	{
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		// 1. Construct neighborgood
		nbh.count = 0;
		nbh.my_type = type[i]-1;
		nbh.types.clear();
		nbh.inds.clear();
		nbh.vecs.clear();
		nbh.dists.clear();

		for (int jj=0; jj<jnum; jj++) 
		{
			j = jlist[jj];
			j &= NEIGHMASK;

			delx = x[j][0] - xtmp;
			dely = x[j][1] - ytmp;
			delz = x[j][2] - ztmp;
			r = sqrt(delx*delx + dely*dely + delz*delz);

			if (r < cutoff) 
			{
				nbh.count++;
				nbh.inds.emplace_back(j);
				nbh.vecs.emplace_back(delx,dely,delz);
				nbh.dists.emplace_back(r);
				nbh.types.emplace_back(type[j]-1);
			}
		}

		// 2. Calculate site energy and their derivatives
		CalcSiteEnergyDers(nbh);
		
		p_site_energy_ders = &buff_site_energy_ders_[0][0];
		energy += buff_site_energy_;
		//if (p_site_en != nullptr)
		//	p_site_en[i] = buff_site_energy_;

		// 3. Add site energy derivatives to force array
		for (int jj=0; jj<nbh.count; jj++)
		{
			int j = nbh.inds[jj];

			f[i][0] += p_site_energy_ders[3*jj+0];
			f[i][1] += p_site_energy_ders[3*jj+1];
			f[i][2] += p_site_energy_ders[3*jj+2];
			
			f[j][0] -= p_site_energy_ders[3*jj+0];
			f[j][1] -= p_site_energy_ders[3*jj+1];
			f[j][2] -= p_site_energy_ders[3*jj+2];
		}

		// 4. Calculate virial stresses per atom (if required)
		/*if (p_site_virial != nullptr) 
			for (int jj = 0; jj < nbh.count; jj++)
			{
				p_site_virial[i][0] -= p_site_energy_ders[3*jj+0] * nbh.vecs[jj][0];
				p_site_virial[i][1] -= p_site_energy_ders[3*jj+1] * nbh.vecs[jj][1];
				p_site_virial[i][2] -= p_site_energy_ders[3*jj+2] * nbh.vecs[jj][2];
				p_site_virial[i][3] -= 0.5 * (p_site_energy_ders[3*jj+1] * nbh.vecs[jj][0] +
											p_site_energy_ders[3*jj+0] * nbh.vecs[jj][1]);
				p_site_virial[i][4] -= 0.5 * (p_site_energy_ders[3*jj+2] * nbh.vecs[jj][0] +
											p_site_energy_ders[3*jj+0] * nbh.vecs[jj][2]);
				p_site_virial[i][5] -= 0.5 * (p_site_energy_ders[3*jj+2] * nbh.vecs[jj][1] +
											p_site_energy_ders[3*jj+1] * nbh.vecs[jj][2]);
			}*/
	}

	if (eflag_global) eng_vdwl += energy;
	//if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMLIP::allocate()
{
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  allocated = 1;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMLIP::settings(int narg, char **arg)
{
  if (narg != 1 && narg != 2) 
    error->all(FLERR, "Illegal pair_style command");

  if (strlen(arg[0]) > 999)
    error->all(FLERR, "MLIP settings file name is too long");

  strcpy(MLIPsettings_filename, arg[0]);
}

/* ----------------------------------------------------------------------
   set flags for type pairs
------------------------------------------------------------------------- */

void PairMLIP::coeff(int narg, char **arg)
{
  if (strcmp(arg[0],"*") || strcmp(arg[1],"*") )
    error->all(FLERR, "Incorrect args for pair coefficients");

  if (!allocated) allocate();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMLIP::init_style()
{
  if (force->newton_pair != 1)
      error->all(FLERR, "Pair style MLIP requires Newton pair on");
  
  Load(MLIPsettings_filename);
  cutoff = p_RadialBasis->max_dist;

  cutoffsq = cutoff*cutoff;
  int n = atom->ntypes;
  for (int i=1; i<=n; i++)
    for (int j=1; j<=n; j++)
      cutsq[i][j] = cutoffsq;

  if (comm->nprocs != 1)
    error->all(FLERR, "MLIP settings are incompatible with parallel LAMMPS mode");

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMLIP::init_one(int i, int j)
{
  return cutoff;
}


void PairMLIP::Load(const string& filename)
{
	alpha_count = 0;

	ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Cannot open " + filename);


	char tmpline[1000];
	string tmpstr;

	ifs.getline(tmpline, 1000);
	int len = (int)((string)tmpline).length();
	if (tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
		tmpline[len - 1] = '\0';

	if ((string)tmpline != "MTP")
		ERROR("Can read only MTP format potentials");

		// version reading block
		ifs.getline(tmpline, 1000);
		len = (int)((string)tmpline).length();
		if (tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
			tmpline[len - 1] = '\0';
		if ((string)tmpline != "version = 1.1.0")
			ERROR("MTP file must have version \"1.1.0\"");

		// name/description reading block
		ifs >> tmpstr;
		if (tmpstr == "potential_name") // optional 
		{
			ifs.ignore(2);
			ifs >> pot_desc;
			ifs >> tmpstr;
		}

		if (tmpstr == "scaling") // optional 
		{
			ifs.ignore(2);
			ifs >> scaling;
			ifs >> tmpstr;
		}

	if (tmpstr != "species_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> species_count;

	ifs >> tmpstr;
	if (tmpstr == "potential_tag")
	{
		getline(ifs, tmpstr);
		ifs >> tmpstr;
	}

	if (tmpstr != "radial_basis_type")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> rbasis_type;
	
	if (rbasis_type == "RBChebyshev")
		p_RadialBasis = new RadialBasis_Chebyshev(ifs);
	else if (rbasis_type == "RBChebyshev_repuls")
		p_RadialBasis = new RadialBasis_Chebyshev_repuls(ifs);
	else if (rbasis_type == "RBShapeev")
		p_RadialBasis = new RadialBasis_Shapeev(ifs);
	else if (rbasis_type == "RBTaylor")
		p_RadialBasis = new RadialBasis_Taylor(ifs);
	else
		ERROR("Wrong radial basis type");

	// We do not need double scaling
	if (p_RadialBasis->scaling != 1.0) {
		scaling *= p_RadialBasis->scaling;
		p_RadialBasis->scaling = 1.0;
	}


	ifs >> tmpstr;
	if (tmpstr != "radial_funcs_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> radial_func_count;

	//Radial coeffs initialization
	int pairs_count = species_count*species_count;           //number of species pairs

	char foo = ' ';

	regression_coeffs.resize(pairs_count*radial_func_count*(p_RadialBasis->rb_size));

	ifs >> tmpstr;
	if (tmpstr == "radial_coeffs")
	{

		for (int s1 = 0; s1 < species_count; s1++)
			for (int s2 = 0; s2 < species_count; s2++)
			{
				ifs >> foo >> foo >> foo;

				double t;

				for (int i = 0; i < radial_func_count; i++)
				{
					ifs >> foo;
					for (int j = 0; j < p_RadialBasis->rb_size; j++)
					{
						ifs >> t >> foo;
						regression_coeffs[(s1*species_count+s2)*radial_func_count*(p_RadialBasis->rb_size) +
							i*(p_RadialBasis->rb_size) + j] = t;

					}

				}

			}

		ifs >> tmpstr;

	}
	else
	{
		//cout << "Radial coeffs not found, initializing defaults" << endl;

		regression_coeffs.resize(species_count*species_count*radial_func_count*(p_RadialBasis->rb_size));


		for (pairs_count = 0; pairs_count < species_count*species_count; pairs_count++)
			for (int i = 0; i < radial_func_count; i++)
			{

				for (int j = 0; j < p_RadialBasis->rb_size; j++)
					regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + j] = 1e-6;

				regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + min(i, p_RadialBasis->rb_size)] = 1e-3;


			}
	}

	if (tmpstr != "alpha_moments_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_moments_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_basic_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_basic = new int[alpha_index_basic_count][4];	
	if (alpha_index_basic == nullptr)
		ERROR("Memory allocation error");

	int radial_func_max = -1;
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_basic[i][0] >> tmpch >> alpha_index_basic[i][1] >> tmpch >> alpha_index_basic[i][2] >> tmpch >> alpha_index_basic[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");

		if (alpha_index_basic[i][0]>radial_func_max)
			radial_func_max = alpha_index_basic[i][0];
	}
	

	//cout << radial_func_count << endl;

	if (radial_func_max!=radial_func_count-1)
		ERROR("Wrong number of radial functions specified");

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_times_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_times = new int[alpha_index_times_count][4];	
	if (alpha_index_times == nullptr)
		ERROR("Memory allocation error");

	for (int i = 0; i < alpha_index_times_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_times[i][0] >> tmpch >> alpha_index_times[i][1] >> tmpch >> alpha_index_times[i][2] >> tmpch >> alpha_index_times[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (tmpstr != "alpha_scalar_moments")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_scalar_moments;
	if (alpha_index_times_count < 0)
		ERROR("Error reading .mtp file");

	alpha_moment_mapping = new int[alpha_scalar_moments];
	if (alpha_moment_mapping == nullptr)
		ERROR("Memory allocation error");

	ifs >> tmpstr;
	if (tmpstr != "alpha_moment_mapping")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);
	for (int i = 0; i < alpha_scalar_moments; i++)
	{
		char tmpch = ' ';
		ifs >> alpha_moment_mapping[i] >> tmpch;
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}
	ifs.ignore(1000, '\n');

	alpha_count = alpha_scalar_moments + 1;


	//Reading linear coeffs
	ifs >> tmpstr;
	if (tmpstr != "species_coeffs")
	{
		//cout << "Linear coeffs not found, initializing defaults, species_count = " << species_count << endl;
		linear_coeffs.resize(alpha_count + species_count - 1);
		for (int i = 0; i < alpha_count + species_count - 1; i++)
			linear_coeffs[i] = 1e-3;
	}
	else
	{
		ifs.ignore(4);

		linear_coeffs.resize(species_count);
		for (int i = 0; i < species_count; i++)
			ifs >> linear_coeffs[i] >> foo;


		ifs >> tmpstr;

		if (tmpstr != "moment_coeffs")
			ERROR("Cannot read linear coeffs");

		ifs.ignore(2);

		linear_coeffs.resize(alpha_count + species_count - 1);

		ifs.ignore(10, '{');


		for (int i = 0; i < alpha_count - 1; i++)
			ifs >> linear_coeffs[i + species_count] >> foo;

	}
		MemAlloc();
		DistributeCoeffs();
}


void PairMLIP::MemAlloc()
{
	int n = alpha_count - 1 + species_count;

	//energy_cmpnts = new double[n];
	//forces_cmpnts.reserve(n * 3);
	//stress_cmpnts = (double(*)[3][3])malloc(n * sizeof(stress_cmpnts[0]));

	moment_vals = new double[alpha_moments_count];
	//basis_vals = new double[alpha_count];
	site_energy_ders_wrt_moments_.resize(alpha_moments_count);

}


void PairMLIP::CalcSiteEnergyDers(const Neighborhood& nbh)
{
	buff_site_energy_ = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);

	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;  //number of Chebyshev polynomials constituting one radial function


	linear_coeffs = LinCoeff();


	if (nbh.count != moment_jacobian_.size2)
		moment_jacobian_.resize(alpha_index_basic_count, nbh.count, 3);

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	moment_jacobian_.set(0);


	// max_alpha_index_basic calculation
	int max_alpha_index_basic = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic = max(max_alpha_index_basic,
			alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
	max_alpha_index_basic++;
	dist_powers_.resize(max_alpha_index_basic);
	coords_powers_.resize(max_alpha_index_basic);

	int type_central = nbh.my_type;

	if (type_central>=species_count)
		printf("Too few species count in the MTP potential!");


	for (int j = 0; j < nbh.count; j++) {
		const Vector3& NeighbVect_j = nbh.vecs[j];

		// calculates vals and ders for j-th atom in the neighborhood
		p_RadialBasis->RB_Calc(nbh.dists[j]);
		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
			p_RadialBasis->rb_vals[xi] *= scaling;
		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
			p_RadialBasis->rb_ders[xi] *= scaling;

		dist_powers_[0] = 1;
		coords_powers_[0] = Vector3(1, 1, 1);
		for (int k = 1; k < max_alpha_index_basic; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
			for (int a = 0; a < 3; a++)
				coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
		}

		int type_outer = nbh.types[j];

		for (int i = 0; i < alpha_index_basic_count; i++) {

			double val = 0, der = 0;
			int mu = alpha_index_basic[i][0];

			for (int xi = 0; xi < R; xi++)
			{
				val += regression_coeffs[(type_central*C + type_outer)*K*R + mu * R + xi] * p_RadialBasis->rb_vals[xi];
				der += regression_coeffs[(type_central*C + type_outer)*K*R + mu * R + xi] * p_RadialBasis->rb_ders[xi];

			}



			int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / nbh.dists[j];

			double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
			double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
			double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / nbh.dists[j];
			moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];

			if (alpha_index_basic[i][1] != 0) {
				moment_jacobian_(i, j, 0) += val * alpha_index_basic[i][1]
					* coords_powers_[alpha_index_basic[i][1] - 1][0]
					* pow1
					* pow2;
			}
			if (alpha_index_basic[i][2] != 0) {
				moment_jacobian_(i, j, 1) += val * alpha_index_basic[i][2]
					* pow0
					* coords_powers_[alpha_index_basic[i][2] - 1][1]
					* pow2;
			}
			if (alpha_index_basic[i][3] != 0) {
				moment_jacobian_(i, j, 2) += val * alpha_index_basic[i][3]
					* pow0
					* pow1
					* coords_powers_[alpha_index_basic[i][3] - 1][2];
			}
		}
	
		//Repulsive term
		//if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls")
		if (nbh.dists[j] < p_RadialBasis->min_dist)
		{
			double multiplier = 10000;
			buff_site_energy_ += multiplier*(exp(-10*(nbh.dists[j]-1)) - exp(-10*(p_RadialBasis->min_dist-1)));
			for (int a = 0; a < 3; a++)
				buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (nbh.dists[j] - 1))/ nbh.dists[j])*nbh.vecs[j][a];
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];
		moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
	}


	// convolving with coefficients
	buff_site_energy_ += linear_coeffs[nbh.my_type];


	for (int i = 0; i < alpha_scalar_moments; i++)
		buff_site_energy_ += linear_coeffs[species_count + i]*linear_mults[i] * moment_vals[alpha_moment_mapping[i]];



	// Backpropagation starts

	// Backpropagation step 1: site energy derivative is the corresponding linear combination
	memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));

	for (int i = 0; i < alpha_scalar_moments; i++)
		site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs[species_count + i]*linear_mults[i];

	// SAME BUT UNSAFE:
	// memcpy(&site_energy_ders_wrt_moments_[0], &basis_coeffs[1],
	//		alpha_scalar_moments * sizeof(site_energy_ders_wrt_moments_[0]));

	// Backpropagation step 2: expressing through basic moments:
	for (int i = alpha_index_times_count - 1; i >= 0; i--) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];

		site_energy_ders_wrt_moments_[alpha_index_times[i][1]] +=
			site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
			* val2 * val0;
		site_energy_ders_wrt_moments_[alpha_index_times[i][0]] +=
			site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
			* val2 * val1;
	}

	// Backpropagation step 3: multiply by the Jacobian:
	for (int i = 0; i < alpha_index_basic_count; i++)
		for (int j = 0; j < nbh.count; j++)
			for (int a = 0; a < 3; a++)
				buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * moment_jacobian_(i, j, a);
}

std::vector<double> PairMLIP::LinCoeff()								//returns linear coefficients
{
	int Rsize = p_RadialBasis->rb_size * species_count*species_count*radial_func_count;
	for (int i = Rsize; i < Rsize + alpha_count + species_count - 1;i++)
	{
		linear_coeffs[i-Rsize]=regression_coeffs[i];
	}
	return linear_coeffs;

}

void PairMLIP::DistributeCoeffs()									//Combine radial and linear coefficients in one array
{
	int radial_size = (int)regression_coeffs.size();
	int max_comp = species_count - 1;				//maximum index of component

	regression_coeffs.resize(radial_size + alpha_count + max_comp);


	if (linear_coeffs.size() == alpha_count)
	{
		for (int i = 0; i <= max_comp; i++)
			regression_coeffs[radial_size + i] = linear_coeffs[0];				//constants for component's site energy shift

		for (int i = 1; i < alpha_count; i++)
			regression_coeffs[radial_size + i + max_comp] = linear_coeffs[i];

	}
	else
		for (int i = 0; i < (int)linear_coeffs.size(); i++)
			regression_coeffs[radial_size + i] = linear_coeffs[i];

	linear_mults.resize(alpha_scalar_moments);
	//max_linear.resize(alpha_scalar_moments);

	for (int i=0;i<alpha_scalar_moments;i++)
	{
		linear_mults[i]=1;
		//max_linear[i]=1e-10;
	}
	
}