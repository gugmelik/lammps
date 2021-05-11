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


#define MAXLINE 1024
#define NEIGHMASK 0x3FFFFFFF
#define DEFAULTCUTOFF 5.0


/* ---------------------------------------------------------------------- */

PairMLIP::PairMLIP(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;

  single_enable = 0;

  inited = false;
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
  
  if (inited) MLIP_finalize();
}

/* ---------------------------------------------------------------------- */

void PairMLIP::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

  // nbh version

  double energy = 0;
  double *p_site_en = NULL;
  double **p_site_virial = NULL;
  if (eflag_atom) p_site_en = &eatom[0];
  if (vflag_atom) p_site_virial = vatom;

  MLIP_calc_nbh(list->inum, 
	  list->ilist, 
	  list->numneigh, 
	  list->firstneigh,
                atom->nlocal,
	  atom->nghost,
	  atom->x, 
	  atom->type,
	  atom->f, 
	  energy,
	  p_site_en,      // if NULL no site energy is calculated
	  p_site_virial); // if NULL no virial stress per atom is calculated

  if (eflag_global) eng_vdwl += energy;
  if (vflag_fdotr) virial_fdotr_compute();
  
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

  if (inited)
    MLIP_finalize();
  
  MLIP_init(MLIPsettings_filename, atom->ntypes, cutoff);

  cutoffsq = cutoff*cutoff;
  int n = atom->ntypes;
  for (int i=1; i<=n; i++)
    for (int j=1; j<=n; j++)
      cutsq[i][j] = cutoffsq;

  if (comm->nprocs != 1)
    error->all(FLERR, "MLIP settings are incompatible with parallel LAMMPS mode");
  inited = true;

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

void PairMLIP::MLIP_init(const char * settings_filename,	// settings filename
			   int ntypes,					// Number of atom types 
			   double& rcut)				// MLIP's cutoff radius returned to driver 
{
  if (p_mlip != nullptr)
		delete p_mlip;

  p_mlip = new MLMTPR(std::string{settings_filename});
  //p_mlip->Load(std::string{settings_filename});

  rcut = p_mlip->CutOff();
}

void PairMLIP::MLIP_finalize()
{
	
	delete p_mlip;
	p_mlip = nullptr;
}


void PairMLIP::MLIP_calc_nbh(int inum,           // input parameter: number of neighborhoods
                   int* ilist,         // input parameter: 
                   int* numneigh,      // input parameter: number of neighbors in each neighborhood (inum integer numbers)
                   int** firstneigh,   // input parameter: pointer to the first neighbor
                   int n_local_atoms,  // input parameter: number of local atoms
                   int n_ghost_atoms,  // input parameter: number of ghost atoms
                   double** x,         // input parameter: array of coordinates of atoms
                   int* types,         // input parameter: array of atom types (inum of integer numbers)
                   double** f,                    // output parameter: forces on atoms (cartesian, n x 3 double numbers)
                   double& en,                    // output parameter: summ of site energies 
                   double* site_en=nullptr,       // output parameter: array of site energies (inum double numbers). if =nullptr while call no site energy calculation is done
                   double** site_virial=nullptr)  // output parameter: array of site energies (inum double numbers). if =nullptr while call no virial-stress-per-atom calculation is done
{
	Neighborhood nbh;

	for (int ii = 0; ii < inum; ii++) 
	{
		int i = ilist[ii];
		double xtmp = x[i][0];
		double ytmp = x[i][1];
		double ztmp = x[i][2];
		int* jlist = firstneigh[i];
		int jnum = numneigh[i];

		// 1. Construct neighborgood
		nbh.count = 0;
		nbh.my_type = types[i]-1;
		nbh.types.clear();
		nbh.inds.clear();
		nbh.vecs.clear();
		nbh.dists.clear();

		for (int jj=0; jj<jnum; jj++) 
		{
			int j = jlist[jj];
			j &= NEIGHMASK;

			double delx = x[j][0] - xtmp;
			double dely = x[j][1] - ytmp;
			double delz = x[j][2] - ztmp;
			double r = sqrt(delx*delx + dely*dely + delz*delz);

			if (r < cutoffsq) 
			{
				nbh.count++;
				nbh.inds.emplace_back(j);
				nbh.vecs.emplace_back(delx,dely,delz);
				nbh.dists.emplace_back(r);
				nbh.types.emplace_back(types[j]-1);
			}
		}

		// 2. Calculate site energy and their derivatives
		p_mlip->CalcSiteEnergyDers(nbh);
		
		double* p_site_energy_ders = &p_mlip->buff_site_energy_ders_[0][0];
		en += p_mlip->buff_site_energy_;
		if (site_en != nullptr)
			site_en[i] = p_mlip->buff_site_energy_;

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
		if (site_virial != nullptr) 
			for (int jj = 0; jj < nbh.count; jj++)
			{
				site_virial[i][0] -= p_site_energy_ders[3*jj+0] * nbh.vecs[jj][0];
				site_virial[i][1] -= p_site_energy_ders[3*jj+1] * nbh.vecs[jj][1];
				site_virial[i][2] -= p_site_energy_ders[3*jj+2] * nbh.vecs[jj][2];
				site_virial[i][3] -= 0.5 * (p_site_energy_ders[3*jj+1] * nbh.vecs[jj][0] +
											p_site_energy_ders[3*jj+0] * nbh.vecs[jj][1]);
				site_virial[i][4] -= 0.5 * (p_site_energy_ders[3*jj+2] * nbh.vecs[jj][0] +
											p_site_energy_ders[3*jj+0] * nbh.vecs[jj][2]);
				site_virial[i][5] -= 0.5 * (p_site_energy_ders[3*jj+2] * nbh.vecs[jj][1] +
											p_site_energy_ders[3*jj+1] * nbh.vecs[jj][2]);
			}
	}
}