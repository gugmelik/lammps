What does a LAMMPS version mean
-------------------------------

The LAMMPS "version" is the date when it was released, such as 1 May
2014. LAMMPS is updated continuously.  Whenever we fix a bug or add a
feature, we release it in the next *patch* release, which are
typically made every couple of weeks.  Info on patch releases are on
`this website page <https://www.lammps.org/bug.html>`_. Every few
months, the latest patch release is subjected to more thorough testing
and labeled as a *stable* version.

Each version of LAMMPS contains all the features and bug-fixes up to
and including its version date.

The version date is printed to the screen and logfile every time you
run LAMMPS. It is also in the file src/version.h and in the LAMMPS
directory name created when you unpack a tarball.  And it is on the
first page of the :doc:`manual <Manual>`.

* If you browse the HTML pages on the LAMMPS WWW site, they always
  describe the most current patch release of LAMMPS.
* If you browse the HTML pages included in your tarball, they
  describe the version you have, which may be older.
