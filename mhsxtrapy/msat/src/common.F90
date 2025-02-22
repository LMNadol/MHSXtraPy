module common
  ! common functions, subroutine and variables for all programs MSAT
  use params

  implicit none

  integer(int32) :: nx, ny, nz
  real(np), allocatable :: bgrid(:, :, :, :)
  real(np) :: xmin, xmax, ymin, ymax, zmin, zmax
  real(np), dimension(:), allocatable :: x, y, z
  integer(int32) :: nnulls
  real(np) :: dx, dy, dz
  real(np), allocatable, dimension(:,:) :: rnulls, rnullsalt, rnullsreal
  real(np), allocatable, dimension(:,:) :: spines, fans
  integer(int32), allocatable, dimension(:) :: signs
  
  real(np), parameter :: pi = acos(-1.0_np)
  real(np), parameter :: dtor = pi/180.0_np
  character(:), allocatable :: filein, fileout

  integer(int64) :: tstart, tstop, count_rate ! to time programs

  contains

    subroutine filenames
      ! sets the filenames as required for reading from and writing to

      character(100) :: arg
      character(:), allocatable :: outname
      integer(int32) :: iarg, ic

      ic = 0
      outname = default_output

      if (command_argument_count() > 0) then
        do iarg = 1, command_argument_count()
          call get_command_argument(iarg,arg)
          if (trim(arg) == '-i') then
            call get_command_argument(iarg+1,arg)
            filein = trim(arg)
            ic = 1
          elseif (trim(arg) == '-o') then
            call get_command_argument(iarg+1,arg)
            deallocate(outname)
            outname = trim(arg)
          elseif (trim(arg) == '--params') then
            call print_params
            stop
          endif
        enddo
      endif
      if (ic == 0) stop 'No input file provided'
      
      fileout = filein(1:index(filein(1:index(filein, '/', .true.)-1), '/', .true.)) &
        //trim(outname)//'/' &
        //trim(filein(index(filein, '/', .true.)+1:index(filein, '.dat', .true.)-1))
      
    end

    subroutine print_params
      ! allows the user to find out what parameters have been set in params.f90 when executable was compiled

      print*, 'GLOBAL PARAMETERS:'
      print*, 'np =', np
      print*, 'nproc =', nproc
      print*, ''
      print*, 'NF PARAMETERS:'
      print*, 'zero =', zero
      print*, 'sig_figs =', sig_figs
      print*, 'boundary_nulls = ', boundary_nulls
      print*, 'gridpoint_nulls = ', gridpoint_nulls
      print*, ''
      print*, 'SF PARAMETERS:'
      print*, 'rspherefact =', rspherefact
      print*, 'nphi =', nphi
      print*, 'ntheta =', ntheta
      print*, 'maxiter =', maxiter
      print*, 'dist_mult =', dist_mult
      print*, ''
      print*, 'SSF PARAMETERS:'
      print*, 'nstart =', nstart
      print*, 'start_dist =', start_dist
      print*, 'ringsmax =', ringsmax
      print*, 'pointsmax =', pointsmax
      print*, 'samemax =', samemax
      print*, 'stepsize =', stepsize
      print*, 'tol =', tol
      print*, 'stepmin =', stepmin 
      print*, 'restart =', restart
      print*, 'nullrestart =', nullrestart
      print*, 'nskip =', nskip
      print*, 'bytesize =', bytesize
      print*, 'assoc_output =', assoc_output
      print*, 'adjust_cartesian_periodicity =', adjust_cartesian_periodicity
      print*, 'adjust_cylindrical_periodicity =', adjust_cylindrical_periodicity
      print*, 'adjust_spherical_periodicity =', adjust_spherical_periodicity
      print*, 'periodic_x =', periodic_x
      print*, 'periodic_y =', periodic_y
      print*, 'periodic_z =', periodic_z
      print*, 'periodic_theta =', periodic_theta
      print*, 'periodic_phi =', periodic_phi
      print*, 'one_sep_per_ring =', one_sep_per_ring

    end

    !********************************************************************************

    function trilinear(r, b)
      ! find the value of vector field at r using the trilinear method

      real(np) :: trilinear(3)
      real(np), allocatable :: b(:, :, :, :)
      real(np) :: r(3)
      real(np) :: cube(2, 2, 2), square(2, 2), line(2)
      real(np) :: x, y, z, xp, yp, zp
      integer(int32) :: nx, ny, nz
      integer(int32) :: dims

#if ssf_code
      call edgecheck(r)
#endif

      xp = r(1)
      yp = r(2)
      zp = r(3)

      nx = floor(xp)
      ny = floor(yp)
      nz = floor(zp)

#if ssf_code
      ! if point goes out of an edge which is not periodic, set point to be at boundary to trilinear
      ! it will go out and be removed soon
      if (outedge(r)) then
        ! first coordinate needs checking in all coordinate systems
        if (xp <= xmin) then
          xp = xmin
          nx = nint(xp)
        elseif (xp >= xmax) then
          xp = xmax
          nx = nint(xp)-1
        endif
#if cartesian
        ! for cartesians
        if (yp <= ymin) then
          yp = ymin
          ny = nint(yp)
        elseif (yp >= ymax) then
          yp = ymax
          ny = nint(yp)-1
        endif
        if (zp <= zmin) then
          zp = zmin
          nz = nint(zp)
        elseif (zp >= zmax) then
          zp = zmax
          nz = nint(zp)-1
        endif
#elif cylindrical
        ! for cylindricals
        if (zp <= zmin) then
          zp = zmin
          nz = nint(zp)
        elseif (zp >= zmax) then
          zp = zmax
          nz = nint(zp)-1
        endif
#endif
      endif
#else
      if (xp >= xmax) then
        xp = xmax
        nx = nint(xp)-1
      endif
      if (yp >= ymax) then
        yp = ymax
        ny = nint(yp)-1
      endif
      if (zp >= zmax) then
        zp = zmax
        nz = nint(zp)-1
      endif
#endif

      x = xp - nx
      y = yp - ny
      z = zp - nz

      do dims = 1, 3
        cube = b(nx:nx+1, ny:ny+1, nz:nz+1, dims)
        square = (1 - z)*cube(:, :, 1) + z*cube(:, :, 2)
        line = (1 - y)*square(:, 1) + y*square(:, 2)
        trilinear(dims) = (1 - x)*line(1) + x*line(2)
      enddo

    end

    !********************************************************************************

    function dot(a, b)
      ! dot product between a and b

      real(np), dimension(3) :: a, b
      real(np) :: dot

      dot = sum(a*b)

    end

    !********************************************************************************

    function cross(a, b)
      ! cross product between a and b

      real(np), dimension(3) :: a, b
      real(np), dimension(3) :: cross

      cross(1) = a(2)*b(3) - a(3)*b(2)
      cross(2) = a(3)*b(1) - a(1)*b(3)
      cross(3) = a(1)*b(2) - a(2)*b(1)

    end

    !********************************************************************************

    function normalise(a)
      ! finds the unit vector of a

      real(np), dimension(3) :: a
      real(np), dimension(3) :: normalise

      normalise = a/modulus(a)

    end

    !********************************************************************************

    subroutine add_vector(x, vec, pos)
      ! adds an row (vec) to a nx column by ny row array at row number pos (or end)

      real(np), allocatable, dimension(:, :) :: x, dummy
      real(np), allocatable, dimension(:) :: vec1
      real(np) :: vec(:)
      integer(int32), optional :: pos
      integer(int32) :: nx, ny, position

      vec1 = vec

      nx = size(x, 1)
      ny = size(x, 2)
      
      if (present(pos)) then
        position = pos
      else
        position = ny+1
      endif

      call move_alloc(x, dummy)

      allocate(x(nx, ny+1))

      x(:, 1:position-1) = dummy(:, 1:position-1)
      x(:, position) = vec1
      x(:, position+1:ny+1) = dummy(:, position:ny)

    end

    !********************************************************************************

    subroutine add_element(x, number, pos)
      ! adds an row with number to a nx array at row number pos (or end)
      
      integer(int32), allocatable, dimension(:) :: x, dummy
      integer(int32), optional :: pos
      integer(int32) :: nx, position, number, flag
      
      flag = number

      nx = size(x, 1)

      if (present(pos)) then
        position = pos
      else
        position = nx+1
      endif

      call move_alloc(x, dummy)

      allocate(x(nx+1))

      x(1:position-1) = dummy(1:position-1)
      x(position) = flag
      x(position+1:nx+1) = dummy(position:nx)

    end

    !********************************************************************************

    subroutine remove_vector(x, pos)
      ! removes row number pos from a 2D array

      real(np), allocatable, dimension(:, :) :: x, dummy
      integer(int32) :: pos, nx, ny

      nx = size(x, 1)
      ny = size(x, 2)

      call move_alloc(x, dummy)

      allocate(x(nx, ny-1))

      x(:, 1:pos-1) = dummy(:, 1:pos-1)
      x(:, pos:ny-1) = dummy(:, pos+1:ny)
      
    end

    !********************************************************************************

    subroutine remove_element(x, pos)
      ! removes row number pos from an 1D integer array

      integer(int32), allocatable, dimension(:) :: x, dummy
      integer(int32) :: pos, nx

      nx = size(x)

      call move_alloc(x, dummy)

      allocate(x(nx-1))

      x(1:pos-1) = dummy(1:pos-1)
      x(pos:nx-1) = dummy(pos+1:nx)
      
    end

    !********************************************************************************

    function modulus(a)
      ! calculates the magnitude a vector a
      
      real(np), dimension(3) :: a
      real(np) :: modulus

      modulus = sqrt(sum(a**2))

    end function

    !********************************************************************************

    function dist(a, b)
      ! calculates the distance between two points in grid units
      
      real(np) :: dist
      real(np), dimension(3) :: a, b

      dist = sqrt(sum((b - a)**2))

    end function

    !********************************************************************************

    function outedge(r)
      ! determines if the point r is outwith the computation box across a non periodic
      ! boundary and flags true or false

      real(np) :: r(3)
      logical :: outedge

#if cartesian
      if (adjust_cartesian_periodicity) then
        outedge = .false.
        if (.not. periodic_x) outedge = outedge .or. r(1) >= xmax .or. r(1) <= xmin
        if (.not. periodic_y) outedge = outedge .or. r(2) >= ymax .or. r(2) <= ymin
        if (.not. periodic_z) outedge = outedge .or. r(3) >= zmax .or. r(3) <= zmin
      else
        outedge = &
          r(1) >= xmax .or. r(1) <= xmin .or. &
          r(2) >= ymax .or. r(2) <= ymin .or. &
          r(3) >= zmax .or. r(3) <= zmin
      endif
#endif
#if cylindrical
      outedge = &
        r(1) >= xmax .or. r(1) <= xmin .or. r(3) >= zmax .or. r(3) <= zmin
      if (adjust_cylindrical_periodicity) then
        if (.not. periodic_phi) outedge = outedge .or. r(2) >= ymax .or. r(2) <= ymin
      endif
#endif
#if spherical
      outedge = r(1) >= xmax .or. r(1) <= xmin
      if (adjust_spherical_periodicity) then
        if (.not. periodic_theta) outedge = outedge .or. r(2) >= ymax .or. r(2) <= ymin
        if (.not. periodic_phi) outedge = outedge .or. r(3) >= zmax .or. r(3) <= zmin
      endif
#endif      
      
    end function  

    !********************************************************************************

    subroutine edgecheck(r)
      ! determines if the point r is outwith the computational box across a 
      ! periodic boundary and moves it if necessary

      real(np) :: r(3)

#if cartesian
      if (adjust_cartesian_periodicity) then
        if (periodic_x) then
          if (r(1) <= xmin) r(1) = r(1) + (xmax - xmin)
          if (r(1) >= xmax) r(1) = r(1) - (xmax - xmin)
        endif
        if (periodic_y) then
          if (r(2) <= ymin) r(2) = r(2) + (ymax - ymin)
          if (r(2) >= ymax) r(2) = r(2) - (ymax - ymin)
        endif
        if (periodic_z) then
          if (r(3) <= zmin) r(3) = r(3) + (zmax - zmin)
          if (r(3) >= zmax) r(3) = r(3) - (zmax - zmin)
        endif
      endif
#endif
#if cylindrical
      if (adjust_cylindrical_periodicity) then
        if (periodic_phi) then
          if (r(2) < ymin) r(2) = r(2) + (ymax - ymin)
          if (r(2) > ymax) r(2) = r(2) - (ymax - ymin)
        endif
      else
        if (r(2) < ymin) r(2) = r(2) + (ymax - ymin)
        if (r(2) > ymax) r(2) = r(2) - (ymax - ymin)
      endif
#endif
#if spherical
      if (adjust_spherical_periodicity) then
        if (periodic_theta) then
          if (r(2) < ymin .or. r(2) > ymax) then
            if (r(2) < ymin) r(2) = 2*ymin - r(2)
            if (r(2) > ymax) r(2) = 2*ymax - r(2)
            if (r(3) < (zmax + zmin)/2) then
              r(3) = r(3) + (zmax - zmin)/2
            else
              r(3) = r(3) - (zmax - zmin)/2
            endif
          endif
        endif
        if (periodic_phi) then
          if (r(3) <= zmin) r(3) = r(3) + (zmax - zmin)
          if (r(3) >= zmax) r(3) = r(3) - (zmax - zmin)
        endif
      else
        if (r(2) < ymin .or. r(2) > ymax) then
          if (r(2) < ymin) r(2) = 2*ymin - r(2)
          if (r(2) > ymax) r(2) = 2*ymax - r(2)
          if (r(3) < (zmax + zmin)/2) then
            r(3) = r(3) + (zmax - zmin)/2
          else
            r(3) = r(3) - (zmax - zmin)/2
          endif
        endif
        if (r(3) <= zmin) r(3) = r(3) + (zmax - zmin)
        if (r(3) >= zmax) r(3) = r(3) - (zmax - zmin)
      endif
#endif
          
    end subroutine

    !********************************************************************************

    subroutine get_startpoints(theta, phi, xs, ys, zs)
      ! from the theta, phi coordinates of a fan vector, produces ring of points in the fanplane

        real(np) :: theta, phi
        real(np), dimension(:) :: xs, ys, zs
        integer(int32) :: i, nlines
        real(np) :: dtheta
        real(np) :: r(3)

        nlines = size(xs)

        dtheta = 2.0_np*pi/nlines

        ! generate nlines start points in ring around equator
        do i = 1, nlines
          r(1) = cos(i*dtheta)
          r(2) = sin(i*dtheta)
          r(3) = 0

          r = rotate(r, theta, phi)

          xs(i) = r(1)*start_dist
          ys(i) = r(2)*start_dist
          zs(i) = r(3)*start_dist
        enddo

    end subroutine

    !********************************************************************************

    function rotate(r, theta, phi)
      ! rotates a r by theta about the xz plane then by phi about the xy plane

      real(np), dimension(3) :: r, rotate
      real(np) :: theta, phi
      real(np), dimension(3, 3) :: roty, rotz, rot

      roty(1,1) = cos(-theta)
      roty(1,2) = 0
      roty(1,3) = -sin(-theta)
      
      roty(2,1) = 0
      roty(2,2) = 1
      roty(2,3) = 0
      
      roty(3,1) = sin(-theta)
      roty(3,2) = 0
      roty(3,3) = cos(-theta)

      rotz(1,1) = cos(-phi)
      rotz(1,2) = sin(-phi)
      rotz(1,3) = 0

      rotz(2,1) = -sin(-phi)
      rotz(2,2) = cos(-phi)
      rotz(2,3) = 0

      rotz(3,1) = 0
      rotz(3,2) = 0
      rotz(3,3) = 1

      rot = matmul(rotz, roty)
      rotate = matmul(rot, r)

    end

    !********************************************************************************

    function gtr(r, gx, gy, gz)
      ! converts a list of grid points to real coordinates

      integer(int32) :: ir
      real(np), dimension(:,:), allocatable :: r, gtr
      real(np), dimension(:), allocatable :: gx, gy, gz
      integer(int32), dimension(3) :: ig

      gtr = r

      do ir = 1, size(r, 2)
        ig = floor(r(:, ir))

        gtr(1, ir) = gx(ig(1)) + (r(1, ir) - ig(1))*(gx(ig(1)+1) - gx(ig(1)))
        gtr(2, ir) = gy(ig(2)) + (r(2, ir) - ig(2))*(gy(ig(2)+1) - gy(ig(2)))
        gtr(3, ir) = gz(ig(3)) + (r(3, ir) - ig(3))*(gz(ig(3)+1) - gz(ig(3)))

      enddo

    end function

    !********************************************************************************

    subroutine file_position(nring, nperring, index, a, p)
      ! calculates the position in the temporary files of the rings and specific points

      integer(int64) :: a, p
      integer(int64) :: uptoring
      integer(int32) :: nring, index
      integer(int32) :: nperring(0:)

      ! 1 whole ring contains 3*np vector points and np association points
      uptoring = sum(int(nperring(0:nring-1), int64))*7
      a = uptoring + int((index-1), int64)
      p = uptoring + int(nperring(nring) + (index-1)*6, int64)

      a = a*4 + 1
      p = p*4 + 1

    end subroutine

end module
