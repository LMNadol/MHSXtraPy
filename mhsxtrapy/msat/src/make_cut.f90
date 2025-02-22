module make_cut_mod

  use params
  use common
  use list_mod

  implicit none

  real(np) :: ds, disttol

  real(np) :: n_pln(3), d_pln

  contains

  subroutine sortpoints(points, pordered, exit_status)

    real(np), dimension(:, :), allocatable :: dists
    real(np), dimension(:), allocatable :: ptdists, spinedists
    real(np), dimension(:, :), allocatable :: points, pordered
    real(np), dimension(3) :: diff, pt1, pt2
    real(np) :: mindist, mindistspine

    integer(int32), dimension(2) :: imindists
    integer(int32) :: imindist, iminspinedist
    integer(int32) :: nspines
    integer(int32) :: ip, jp, dir, npoints

    logical :: exit_status

    type(list) :: pt_list

    exit_status = .false.

    ! search for closest points in first 5000 points if large number of points
    if (size(points, 2) > 5000) then
      npoints = 5000
    else
      npoints = size(points, 2)
    endif

    ! calculate point to point distances
    allocate(dists(npoints, npoints))
    dists = 1000 ! maybe needs improving
    do ip = 1, npoints
      dists(ip, ip+1:npoints) = sqrt((points(1, ip) - points(1, ip+1:npoints))**2 &
        + (points(2, ip) - points(2, ip+1:npoints))**2 + (points(3, ip) - points(3, ip+1:npoints))**2)
    enddo
    imindists = minloc(dists)
    
    if (dists(imindists(1), imindists(2)) > disttol*10) then
      ! exit if closest points far away
      exit_status = .true.
    else
      ! add two closest points to list to start the line
      deallocate(dists)
      call pt_list%create()
      call pt_list%append(points(:, imindists(1)))
      call pt_list%append(points(:, imindists(2)))
      call remove_vector(points, imindists(1))
      if (imindists(1) < imindists(2)) then
        call remove_vector(points, imindists(2)-1)
      else
        call remove_vector(points, imindists(2))
      endif

      ! allocate array to store distances between current points and all spines
      nspines = size(spines, 2)
      allocate(spinedists(nspines))

      do dir = 1, 2
        ! add points in the each direction
        do while (size(points, 2) > 0)
          ! allocate array to store distances between current point to all other points
          allocate(ptdists(size(points, 2)))

          ! set pt1/2 to be first two points in list
          pt1 = pt_list%first%r
          pt2 = pt_list%first%next%r

          ! vector connecting pt1/2
          diff = normalise(pt1 - pt2)

          ! required distances to first point
          ptdists = (points(1, :) - pt1(1))**2 + (points(2, :) - pt1(2))**2 + (points(3, :) - pt1(3))**2
          spinedists = (spines(1, :) - pt1(1))**2 + (spines(2, :) - pt1(2))**2 + (spines(3, :) - pt1(3))**2

          ! shortest distance from point to all other points and location in array
          imindist = minloc(ptdists, 1)
          mindist = ptdists(imindist)
          
          ! shortest distance from point to all spines
          if (nspines == 0) then
            mindistspine = 1e200_np
          else
            ! iminspinedist = minloc(spinedists, 1)
            mindistspine = minval(spinedists, 1)
          endif

          deallocate(ptdists, spinedists)

          ! if new point is good, add to line
          if (mindist < mindistspine) then
            if (dot(diff, normalise(points(:, imindist) - pt1)) < -0.98) then ! sharp corner in line
              exit
            elseif (mindist < disttol) then
              call pt_list%prepend(points(:, imindist))
              call remove_vector(points, imindist)
            elseif (dot(diff, normalise(points(:, imindist) - pt1)) > 0.95 .and. &
              mindist < 5*disttol) then ! line is almost straight
              call pt_list%prepend(points(:, imindist))
              call remove_vector(points, imindist)
            else
              exit
            endif
          else
            exit
          endif
        enddo

        ! reverse line so can add to line using loop
        call pt_list%reverse()

      enddo

      pordered = pt_list%to_array()
      call pt_list%destroy()
    endif

  end subroutine

  function plane(r)
    ! function to calculate required plane

    implicit none

    real(np) :: plane, r(3)

    plane = dot(n_pln, r) - d_pln

  end function

  function check_crossing(r1, r2)
    ! checks if line between two points crosses plane

    implicit none

    logical :: check_crossing
    real(np), dimension(3) :: r1, r2

    check_crossing = plane(r1)*plane(r2) <= 0

  end function

  function find_crossing(r1, r2)
    ! finds crossing of plane using linear interpolation of two points

    implicit none

    real(np), dimension(3) :: find_crossing, r1, r2
    real(np) :: s

    s = (d_pln - dot(r1, n_pln))/(dot(r2 - r1, n_pln))
    find_crossing = r1 + s*(r2 - r1)

  end function

end module

!######################################################################################

program make_cut
  ! main make_cut program
  ! finds the intersection of the magnetic skeleton features found by nf, sf, ssf and hcs
  ! and the plane given as input to the program
  ! Execute ./make_cut -i field3d.dat -n 1.0 0.0 0.0 -d 2.0 for the plane
  !   1.0*x + 0.0*y + 0.0*z = 2.0

  use params
  use common
  use make_cut_mod

  implicit none

  real(np) :: r0
  integer(int32) :: pc = 0
  character(:), allocatable :: astr, bstr, cstr, dstr, sign
  character(*), parameter :: fmt = '(A,i3.3,f5.4)', neg = '         ', pos = '        '
  character(:), allocatable :: file_pln

  integer(int64) :: ig
  real(np), dimension(:), allocatable :: xg, yg, zg

  integer(int32) :: iarg, jarg
  character(10) :: arg

  type(list) :: spine_list, ring_list
  integer(int64) :: ia, ip, uptorings, uptoassoc, uptoring1, uptoring2
  integer(int32) :: inull, jnull, iring, iline, nextline, flag, ihcs, ipt, isep
  integer(int32) :: nrings, npoints, nlines, nseps, ncomp, nskip_file

  integer(int32), dimension(:), allocatable :: association, break
  integer(int32), dimension(:), allocatable :: nperring
  real(np), dimension(:, :), allocatable :: line, line2, points, pordered
  real(np) :: s, point(3), hstep
  logical :: iexist, exit_status, do_hcs = .false., do_all_ssf = .true., too_much_memory

  integer(int32) :: nspine, dir

  integer(int32) :: iprint, iorder

  integer(int64) :: num_nums, leftover_num, read_index, read_num
  integer(int64), parameter :: large_read = 500000000_int64
  real(np), dimension(:, :), allocatable :: lines
  
  integer(int32), dimension(:), allocatable :: associations, breaks

  call filenames

  if (command_argument_count() > 0) then
    do iarg = 1, command_argument_count()
      call get_command_argument(iarg,arg)
      if (trim(arg) == '--hcs') then
        do_hcs = .true.
      endif
      if (trim(arg) == '--only-hcs') then
        do_all_ssf = .false.
      endif
      if (trim(arg) == '-n') then
        do jarg = 1, 3
          call get_command_argument(iarg+jarg,arg)
          read(arg,*) n_pln(jarg)
        enddo
        pc = pc + 1
      endif
      if (trim(arg) == '-d') then
        call get_command_argument(iarg+1,arg)
        read(arg,*) d_pln
        pc = pc + 1
      endif
    enddo
  endif
  if (pc < 2) then
    stop "Need to provide normal to the plane -n and planar constant -d"
  endif

  ! set up the part of filename defining the plane
  if (n_pln(1) < 0) then
    sign = '-'
    astr = neg
  else
    sign = ''
    astr = pos
  endif
  write(astr, fmt) sign, int(abs(n_pln(1))), abs(n_pln(1))-int(abs(n_pln(1)))
  if (n_pln(2) < 0) then
    sign = '-'
    bstr = neg
  else
    sign = ''
    bstr = pos
  endif
  write(bstr, fmt) sign, int(abs(n_pln(2))), abs(n_pln(2))-int(abs(n_pln(2)))
  if (n_pln(3) < 0) then
    sign = '-'
    cstr = neg
  else
    sign = ''
    cstr = pos
  endif
  write(cstr, fmt) sign, int(abs(n_pln(3))), abs(n_pln(3))-int(abs(n_pln(3)))
  if (d_pln < 0) then
    sign = '-'
    dstr = neg
  else
    sign = ''
    dstr = pos
  endif
  write(dstr, fmt) sign, int(abs(d_pln)), abs(d_pln)-int(abs(d_pln))

  file_pln = astr // '_' // bstr // '_' // cstr // '_' // dstr

  print*, 'Making cut at in file '// filein // ' using plane:'
  print '(5X, 7A)', &
    astr, ' * x1 + ', bstr, ' * x2 + ', cstr, ' * x3 = ', dstr

  ! read in number of null points - no need for positions
  open(unit=10, file=trim(fileout)//'-nullpos.dat', access='stream', status='old')
    read(10) nnulls
  close(10)

  ! read in only the grid coordinates - no need to magnetic field values here
  open(unit=10, file=filein, access='stream', status='old')
    read(10) nx, ny, nz ! number of vertices
    allocate(xg(nx), yg(ny), zg(nz))
    ig = 3_int64*4_int64 + 3_int64*int(nx, int64)*int(ny, int64)*int(nz, int64)*8_int64 + 1_int64
    read(10, pos=ig) xg, yg, zg
  close(10)

  ! work out how often to print for progress update
  if (nnulls < 10) then
    iprint = 1
  else
    iorder = floor(log10(1.0_np*nnulls))
    iprint = (nnulls/(10**iorder) + 1)*10**(iorder-1)
  endif

  call system_clock(tstart, count_rate) ! to time how long it takes

  ! read in the value of h used during the running of ssf
  open(unit=40, file=trim(fileout)//'-ringinfo.dat', access='stream', status='old')
    read(40) nrings, nrings, nrings, hstep
  close(40)

  ds = maxval([maxval(xg(2:nx) - xg(1:nx-1)), &
               maxval(yg(2:ny) - yg(1:ny-1)), &
               maxval(zg(2:nz) - zg(1:nz-1))])*2*hstep
  disttol = ds/10
  print*, ds, hstep
  !********************************************************************************
    
  ! for each spine, check whether we cross desired plane and add point to file
  print*, 'Spines'
  open(unit=80, file=trim(fileout)//'-spines-cut_'//file_pln//'.dat', access='stream', status='replace')
  open(unit=10, file=trim(fileout)//'-spines.dat', access='stream', status='old')

  call spine_list%create()

  ipt = 0
  write(80) ipt
  
  do inull = 1, nnulls
    ! read in spines for each direction
    do dir = 1, 2
      read(10) nspine
      allocate(line(3, nspine))
      read(10) line
      do iline = 1, nspine-1
        nextline = iline + 1
        if (check_crossing(line(:, iline), line(:, nextline))) then
          if (dist(line(:, iline), line(:, nextline)) < 1) then
            point = find_crossing(line(:, iline), line(:, nextline))
            write(80) inull, point
            ipt = ipt + 1
            call spine_list%append(point)
          endif
        endif
      enddo
      deallocate(line)
    enddo
  enddo
  write(80, pos=1) ipt
  close(80)
  close(10)

  spines = spine_list%to_array()

  !********************************************************************************

  if (do_all_ssf) then
    ! for each separator, check whether we cross desired plane and add point to file
    print*, 'Separators'
    open(unit=80, file=trim(fileout)//'-separators-cut_'//file_pln//'.dat', access='stream', status='replace')
    open(unit=20, file=trim(fileout)//'-connectivity.dat', access='stream', status='old')
    open(unit=30, file=trim(fileout)//'-separators.dat', access='stream', status='old')

    ipt = 0
    write(80) ipt
    do inull = 1, nnulls
      read(20) nseps
      do isep = 1, nseps
        read(20) flag, flag
        read(30) npoints
        allocate(line(3, npoints))
        read(30) line
        do iline = 1, npoints-1
          nextline = iline+1
          if (check_crossing(line(:, iline), line(:, nextline))) then
            if (dist(line(:, iline), line(:, nextline)) < 1) then
              point = find_crossing(line(:, iline), line(:, nextline))
              write(80) inull, flag, point
              ipt = ipt + 1
            endif
          endif
        enddo
        read(20) flag, flag
        deallocate(line)
      enddo
    enddo
    
    write(80, pos=1) ipt
    close(80)
    close(30)
    close(20)

    !********************************************************************************

    ! for each adjacent ring, check whether we cross desired plane and add point to file
    print*, 'Rings'
    open(unit=80, file=trim(fileout)//'-rings-cut_'//file_pln//'.dat', access='stream', status='replace')
    open(unit=40, file=trim(fileout)//'-ringinfo.dat', access='stream', status='old')
    open(unit=50, file=trim(fileout)//'-rings.dat', access='stream', status='old')
    open(unit=60, file=trim(fileout)//'-breaks.dat', access='stream', status='old')
    inquire(file=trim(fileout)//'-assocs.dat', exist=iexist)
    if (iexist) then
      open(unit=55, file=trim(fileout)//'-assocs.dat', access='stream', status='old')
    else
      stop "Associations file does not exist."
    endif

    nlines = 0

    write(80) nlines

    read(40) nrings
    read(40) nskip_file
    allocate(nperring(0:ceiling(real(nrings)/nskip_file-1)))
    read(40) nrings, nrings, nrings ! just read into unneeded variable to reach correct position in file

    uptoassoc = 0
    uptorings = 0
    do inull = 1, nnulls
      
      read(40) nperring
      nrings = count(nperring > 0) - 1 ! index of nperring starts at 0

      ! two options to read in points: one fast and one slow
      ! depends on if all points too big for memory/max fortran array size
      ! if number of points isn't too big... uses fast method
      ! and reads all points in now
      num_nums = sum(int(nperring, int64))
      if (num_nums > 2500000000_int64) then
        too_much_memory = .true.
      else
        too_much_memory = .false.
      endif
      ! print*, num_nums

      if (.not. too_much_memory) then
        ! 500000000 seems to be maximum number of int32 integers fortran can read at a time
        ! don't have concrete evidence for this number... may need changing but seems to work
        allocate(lines(3, num_nums), associations(num_nums), breaks(num_nums))

        ia = uptoassoc + 1
        ip = uptorings + 1
        
        ! if bigger than a certain read size then read in chunks
        ! otherwise read all of it
        if (num_nums > large_read) then
          leftover_num = num_nums
          read_index = 1
          inquire(50, pos=ip)
          inquire(55, pos=ia)
          inquire(60, pos=ia)
          do while (leftover_num > 0)
            if (leftover_num > large_read) then
              read_num = large_read
            else
              read_num = leftover_num
            endif
            read(50) lines(:, read_index:read_index+read_num-1)
            read(55) associations(read_index:read_index+read_num-1)
            read(60) breaks(read_index:read_index+read_num-1)
            read_index = read_index + read_num
            leftover_num = leftover_num - read_num
          enddo
        else
          read(50, pos=ip) lines
          read(55, pos=ia) associations
          read(60, pos=ia) breaks
        endif
      endif
      
      call ring_list%create()

      do iring = nrings, 1, -1

        uptoring1 = sum(int(nperring(0:iring-2), int64))
        uptoring2 = uptoring1 + nperring(iring-1)
        
        ! if have to use the slow method start reading in ring by ring
        ! otherwise use the arrays read in above using fast method
        if (too_much_memory) then
          allocate(association(nperring(iring)), break(nperring(iring)))
          ia = uptoassoc + uptoring2*4_int64 + 1
          read(55, pos=ia) association
          read(60, pos=ia) break
          
          if (iring == nrings) then
            allocate(line(3, nperring(iring)))
            ip = uptorings + uptoring2*24_int64 + 1
            read(50, pos=ip) line
          endif

          allocate(line2(3, nperring(iring-1)))
          ip = uptorings + uptoring1*24_int64 + 1
          read(50, pos=ip) line2
        else
          association = associations(uptoring2+1:uptoring2+nperring(iring))
          break = breaks(uptoring2+1:uptoring2+nperring(iring))
          if (iring == nrings) line = lines(:, uptoring2+1:uptoring2+nperring(iring))
          line2 = lines(:, uptoring1+1:uptoring2)
        endif

        ! find crossing of rings
        ! first by using associations from one ring to next
        ! second by going around rings in order of points
        do iline = 1, nperring(iring)
          if (check_crossing(line(:, iline), line2(:, association(iline)))) then
            if (dist(line(:, iline), line2(:, association(iline))) < 1) then
              ! find point in crossing plane and add to list of points
              point = find_crossing(line(:, iline), line2(:, association(iline)))
              call ring_list%append(point)
            endif
          endif
        enddo
        do iline = 1, nperring(iring)-1
          nextline = iline + 1
          if (check_crossing(line(:, iline), line(:, nextline))) then
            if (break(iline) == 0 .and. dist(line(:, iline), line(:, nextline)) < 1) then
              ! find point in crossing plane and add to list of points
              point = find_crossing(line(:, iline), line(:, nextline))
              call ring_list%append(point)
            endif
          endif
        enddo
        call move_alloc(line2, line)
        deallocate(association, break)
      enddo

      if (.not. too_much_memory) deallocate(lines, associations, breaks)
      if (allocated(line)) deallocate(line)

      ! find new read file points for slow method
      uptoring1 = sum(int(nperring, int64))
      uptoassoc = uptoassoc + uptoring1*4_int64
      uptorings = uptorings + uptoring1*24_int64

      points = ring_list%to_array()
      call ring_list%destroy()

      ! sort points until list is exhausted
      do while (size(points, 2) > 0)

        call sortpoints(points, pordered, exit_status)
        if (exit_status) then
          exit
        else
          write(80) inull, size(pordered, 2), pordered
          nlines = nlines + 1
          deallocate(pordered)
        endif

      enddo
      
      deallocate(points)
      if (mod(inull, iprint) == 0) print*, 'Done', inull, 'of', nnulls
    enddo
    write(80, pos=1) nlines
    close(80)
    close(40)
    close(50)
    close(55)
    deallocate(nperring)

    print*, 'Finished null rings'
  
  endif

  !********************************************************************************

  if (do_hcs) then
    ! same as the rings above but for the hcs rings - see above for comments
    open(unit=80, file=trim(fileout)//'-hcs-cut_'//file_pln//'.dat', access='stream', status='replace')
    open(unit=40, file=trim(fileout)//'-hcs-ringinfo.dat', access='stream', status='old')
    open(unit=50, file=trim(fileout)//'-hcs-rings.dat', access='stream', status='old')
    inquire(file=trim(fileout)//'-hcs-assocs.dat', exist=iexist)
    if (iexist) then
      open(unit=55, file=trim(fileout)//'-hcs-assocs.dat', access='stream', status='old')
    else
      stop "HCS Associations file does not exist."
    endif

    nlines = 0
    write(80) nlines
    
    read(40) nrings
    read(40) nskip_file
    allocate(nperring(0:ceiling(real(nrings)/nskip_file-1)))
    read(40) nrings, nrings, nrings, ncomp
    
    uptoassoc = 0
    uptorings = 0

    ! uptonull = 0
    do ihcs = 1, ncomp
      allocate(points(3, 0))
      call ring_list%create()
      read(40) nperring

      nrings = count(nperring > 0) - 1
      num_nums = sum(int(nperring, int64))
      allocate(lines(3, num_nums), associations(num_nums))
      ia = uptoassoc + 1
      ip = uptorings + 1

      if (num_nums > large_read) then
        leftover_num = num_nums
        read_index = 1
        inquire(50, pos=ip)
        inquire(55, pos=ia)
        do while (leftover_num > 0)
          if (leftover_num > large_read) then
            read_num = large_read
          else
            read_num = leftover_num
          endif
          read(50) lines(:, read_index:read_index+read_num)
          read(55) associations(read_index:read_index+read_num)
          read_index = read_index + read_num
          leftover_num = leftover_num - read_num
        enddo
      else
        read(50, pos=ip) lines
        read(55, pos=ia) associations
      endif

      do iring = nrings, 1, -1

        uptoring1 = sum(int(nperring(0:iring-2), int64))
        uptoring2 = uptoring1 + nperring(iring-1)

        ! ia = uptoassoc + uptoring2*4_int64 + 1
        ! ip = uptorings + uptoring2*24_int64 + 1
      
        ! allocate(association(nperring(iring)), line(3, nperring(iring)))
        ! read(55, pos=ia) association
        ! read(50, pos=ip) line

        ! ip = uptorings + uptoring1*24_int64 + 1

        ! allocate(line2(3,nperring(iring-1)))
        ! read(50, pos=ip) line2

        association = associations(uptoring2+1:uptoring2+nperring(iring))
        if (iring == nrings) line = lines(:, uptoring2+1:uptoring2+nperring(iring))
        line2 = lines(:, uptoring1+1:uptoring2)

        do iline = 1, nperring(iring)
          if (check_crossing(line(:, iline), line2(:, association(iline)))) then
            if (dist(line(:, iline), line2(:, association(iline))) < 1) then
              ! find point in between at r0 and add to line
              point = find_crossing(line(:, iline), line2(:, association(iline)))
              call ring_list%append(point)
            endif
          endif
        enddo
        call move_alloc(line2, line)
        deallocate(association)
      enddo
      deallocate(lines, associations)
      if (allocated(line)) deallocate(line)
      uptoring1 = sum(int(nperring, int64))
      uptoassoc = uptoassoc + uptoring1*4_int64
      uptorings = uptorings + uptoring1*24_int64

      points = ring_list%to_array()
      call ring_list%destroy()

      print*, 'sorting hcs points'
      print*, size(points, 2)

      do while (size(points, 2) > 0)

        call sortpoints(points, pordered, exit_status)
        if (exit_status) then
          exit
        else
          write(80) int(0, int32), size(pordered, 2), pordered
          nlines = nlines + 1
          deallocate(pordered)
        endif

      enddo

      deallocate(points)
    enddo

    write(80, pos=1) nlines
    close(80)
    close(40)
    close(50)
    close(55)
    deallocate(nperring)

  !********************************************************************************

    ! for each hcs separator, check whether we cross the plane and add point to file
    ! need to do this after rings (can't remember why)
    open(unit=80, file=trim(fileout)//'-hcs-separators-cut_'//file_pln//'.dat', access='stream', status='replace')
    open(unit=20, file=trim(fileout)//'-hcs-connectivity.dat', access='stream', status='old')
    open(unit=30, file=trim(fileout)//'-hcs-separators.dat', access='stream', status='old')
    
    ipt = 0
    write(80) ipt
    read(20) nseps
    do isep = 1, nseps
      read(20) inull, flag
      read(30) npoints
      allocate(line(3, npoints))
      read(30) line
      do iline = 1, npoints-1
        nextline = iline+1
        if (check_crossing(line(:, iline), line(:, nextline))) then
          if (dist(line(:, iline), line(:, nextline)) < 1) then
            point = find_crossing(line(:, iline), line(:, nextline))
            write(80) flag, point
            ipt = ipt + 1
          endif
        endif
      enddo
      read(20) flag, flag ! don't care about these - just reposition file marker
      deallocate(line)
    enddo
    
    write(80, pos=1) ipt
    close(80)
    close(30)
    close(20)
  endif

  call system_clock(tstop, count_rate)
  print*, 'Time taken:', dble(tstop - tstart)/dble(count_rate), "(", dble(tstop - tstart)/dble(count_rate)/60, "minutes)"

  call spine_list%destroy()

end program