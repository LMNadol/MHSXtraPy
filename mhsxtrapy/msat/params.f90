module params

  use iso_fortran_env

  implicit none
  ! File containing all changeable parameters in the MSAT package.
  ! It is not recommended to change the majority unless absolutely needed.
  ! Full descriptions of each are available in the MSAT manual

  ! Global Parameters
  ! -----------------
  integer(int32), parameter :: np = real64
  ! Number of threads to be used by OpenMP
  ! set to zero to use the environment variable OMP_NUM_THREADS
  integer(int32), parameter :: nproc = 4
  ! The default directory name to output data to
  character(*), parameter :: default_output = 'output'

  ! Null Finder Parameters
  ! ----------------------
  ! what the code treats as zero
  ! for large regions of weak field, set zero to be a small (but larger) number i.e. 1e-12_np
  real(np), parameter :: zero = 1e-16_np
  ! the number of significant figures of accuracy required for the null
  integer(int32), parameter :: sig_figs = 8
  ! whether to include null points close to the boundary of vector field grids
  ! if true sign finder may struggle to analyse them
  logical, parameter :: boundary_nulls = .false.
  ! whether to check the grid points for null points - may be useful for analytical grids
  logical, parameter :: gridpoint_nulls = .true.

  ! Sign Finder Parameters
  ! ----------------------
  ! radius factor (in grid coordinates) of sphere on which to 
  ! place start points (rsphere = rspherefact*10**-sig_figs)
  real(np), parameter :: rspherefact = 100.0_np
  ! number of start points in phi direction
  integer(int32), parameter :: nphi = 50
  ! number of start points in theta direction
  integer(int32), parameter :: ntheta = nphi/2
  ! maximum number of iterations for convergence method
  integer(int32), parameter :: maxiter = 20000
  ! multiple of rsphere distance to move points at each iteration
  real(np), parameter :: dist_mult = 2e-2_np

  ! Separatrix Surface Finder Parameters
  ! (Most of these also apply to HCS)
  ! ------------------------------------
  ! number of startpoints in ring
  integer(int32), parameter :: nstart = 500
  ! distance from null to start ring
  real(real64), parameter :: start_dist = 0.01_np
  ! maximum number of rings
  integer(int32), parameter :: ringsmax = 50000
  ! maximum number of points in ring
  integer(int32), parameter :: pointsmax = 200000
  ! number of consectutive rings which are the same size which stop ssf
  integer(int32), parameter :: samemax = 1000
  ! step size h after 50 iterations (otherwise 5 times smaller)
  real(np), parameter :: stepsize = 0.1_np
  ! tolerance of rkf45 scheme
  real(np), parameter :: tol = 1e-6_np
  ! minimum step length
  real(np), parameter :: stepmin = 1e-5_np
  ! turn on restart function of the code
  logical, parameter :: restart = .false.
  ! which null to restart from (make sure it's correct)
  ! set to 0 to let the code decide using already written data
  integer(int32), parameter :: nullrestart = 0
  ! number of rings to skip in output to file
  ! set to 1 to write all rings to file
  integer(int32), parameter :: nskip = 1
  ! output bitsize of floating points numbers for rings
  ! default real64 (double precision)
  ! also allows make_cut to read points in correctly
  integer(int32), parameter :: bytesize = real64
  ! whether ssf outputs associations
  ! associations only currently required for make_cut
  logical, parameter :: assoc_output = .true.
  ! allow changes in default periodicity
  ! default is no periodicity in cartesian and
  ! periodic theta, phi in cylindrical/spherical
  logical, parameter :: adjust_cartesian_periodicity = .false.
  logical, parameter :: adjust_cylindrical_periodicity = .false.
  logical, parameter :: adjust_spherical_periodicity = .false.
    ! turn on and off periodicity in x, y, z, theta and phi
    ! these do nothing if adjust_***_periodicities is .false.
    logical, parameter :: periodic_x = .false.
    logical, parameter :: periodic_y = .false.
    logical, parameter :: periodic_z = .false.
    logical, parameter :: periodic_theta = .false.
    logical, parameter :: periodic_phi = .false.
  ! Only allow one separator to be found per null point per ring
  ! This can help to reduce the number of separators found in complex fields
  logical, parameter :: one_sep_per_ring = .false.

end module
