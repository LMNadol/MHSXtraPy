! code to find the separatrix surface and separators of heliospheric current sheet
program hcs

  use omp_lib
  use params
  use common
  use trace
  use ring

  implicit none

  real(np) :: r(3)
  real(np) :: h, h0
  real(np) :: slowdown, slowdist
  integer(int32) :: nearflag

  integer(int32) :: iring, iline, inull, jnull, dir

  integer(int32) :: nlines
  integer(int32) :: nrings
  integer(int32) :: nperring(0:ringsmax)

  logical :: exitcondition, out

  integer(int32) :: ihcs, it, nnt, nnp, imin, jp, nsplit
  integer(int32), dimension(:), allocatable :: npoints
  real(np), dimension(:), allocatable :: tgrid, pgrid, line, diffs, dists
  real(np), dimension(:,:), allocatable :: points, pordered, ind_line
  real(np) :: bvec(3), ds

  ! file writing
  integer(int64) :: ia, ip, uptonullconn
  integer(int32) :: isep, nullnum1, nullnum2, linenum, ringnum, iskip
  real(np), dimension(:,:), allocatable :: rsep, rring, write_ring
  integer(int32), dimension(:), allocatable :: assoc_temp
  character(:), allocatable :: tempfile
  integer(int32) :: nseps1

  print*,'#######################################################################'
  print*,'#                      Separatrix Surface Finder                      #'
  print*,'#######################################################################'

#if _OPENMP
  if (nproc > 0) call omp_set_num_threads(nproc)
  print*, 'Using', nproc, 'processors'
#endif

  call filenames

  ! read in data
  open(unit=10, file=filein, access='stream', status='old')
    read(10) nx, ny, nz ! number of vertices
    allocate(bgrid(nx,ny,nz,3))
    allocate(x(nx), y(ny), z(nz))
    read(10) bgrid(:,:,:,1)
    read(10) bgrid(:,:,:,2)
    read(10) bgrid(:,:,:,3)
    read(10) x, y, z
  close(10)

  xmin = 1
  xmax = nx
  ymin = 1
  ymax = ny
  zmin = 1
  zmax = nz
  print*, "Data read in, grid dimensions are", nx, ny, nz

  call system_clock(tstart,count_rate) ! to time how long it takes

  ! read in null data

  open(unit=10, file=trim(fileout)//'-nullpos.dat', access='stream', status='old')
    read(10) nnulls
    allocate(rnulls(3,nnulls), rnullsreal(3,nnulls))
    read(10) rnulls
    read(10) rnullsreal
  close(10)
  open(unit=10, file=trim(fileout)//'-nulldata.dat', access='stream', status='old')
    read(10) nnulls
    allocate(signs(nnulls),spines(3,nnulls),fans(3,nnulls))
    read(10) signs, spines, fans
  close(10)

  rnullsalt = rnulls

#if spherical
  ! for sphericals
  do inull = 1, nnulls
    ! check whether null is at the lower phi boundary
    if (rnullsalt(3,inull) < zmin + 1) then
      rnullsalt(3,inull) = rnullsalt(3,inull) - 1
      call edgecheck(rnullsalt(:,inull))
      rnullsalt(3,inull) = rnullsalt(3,inull) + 1
    endif

    ! check whether null is at the upper phi boundary
    if (rnullsalt(3,inull) > zmax - 1) then
      rnullsalt(3,inull) = rnullsalt(3,inull) + 1
      call edgecheck(rnullsalt(:,inull))
      rnullsalt(3,inull) = rnullsalt(3,inull) - 1
    endif

    ! check whether null is at the lower theta boundary
    if (rnullsalt(2,inull) < ymin + 1) then
      rnullsalt(2,inull) = rnullsalt(2,inull) - 1
      call edgecheck(rnullsalt(:,inull))
      rnullsalt(2,inull) = rnullsalt(2,inull) - 1
    endif

    ! check whether null is at the upper theta boundary
    if (rnullsalt(2,inull) > ymax - 1) then
      rnullsalt(2,inull) = rnullsalt(2,inull) + 1
      call edgecheck(rnullsalt(:,inull))
      rnullsalt(2,inull) = rnullsalt(2,inull) + 1
    endif
  enddo
#elif cylindrical
  ! for cylindricals
  do inull = 1, nnulls
    ! check whether null is at lower phi boundary
    if (rnullsalt(2,inull) < ymin + 1) then
      rnullsalt(2,inull) = rnullsalt(2,inull) - 1
      call edgecheck(rnullsalt(:,inull))
      rnullsalt(2,inull) = rnullsalt(2,inull) + 1
    endif

    ! check whether null is at upper phi boundary
    if (rnullsalt(2,inull) > ymax - 1) then
      rnullsalt(2,inull) = rnullsalt(2,inull) + 1
      call edgecheck(rnullsalt(:,inull))
      rnullsalt(2,inull) = rnullsalt(2,inull) - 1
    endif
  enddo
#endif

  uptonullconn = 5

  ! stores all info regarding size of rings
  open(unit=10,file=trim(fileout)//'-hcs-ringinfo.dat',status='replace',access='stream')
  ! stores all the coordinates of rings in original coordinate system
  open(unit=20,file=trim(fileout)//'-hcs-rings.dat',status='replace',access='stream')
  ! stores all the associations between rings
  if (assoc_output) open(unit=25, file=trim(fileout)//'-hcs-assocs.dat', status='replace', access='stream')
  ! stores all the breaks on rings
  open(unit=30, file=trim(fileout)//'-hcs-breaks.dat', status='replace', access='stream')
  ! stores all the connection info about each separator
  open(unit=40,file=trim(fileout)//'-hcs-connectivity.dat',status='replace',access='stream')
  ! stores all the coordinates of the separator lines in original coordinate system
  open(unit=50,file=trim(fileout)//'-hcs-separators.dat',status='replace',access='stream')

  ! calculate the points along the hcs line
  nsplit = 10
  nnt = nsplit*(ny-1)+1
  nnp = nsplit*(nz-1)+1

  tgrid = real([(it,it=0,nnt-1)], np)/nsplit + 1
  pgrid = real([(ip,ip=0,nnp-1)], np)/nsplit + 1
  tgrid(1) = tgrid(1) + 1e-6_np
  tgrid(nnt) = tgrid(nnt) - 1e-6_np

  allocate(points(3, 0))
  do it = 1, nnt

    allocate(line(nnp))

    r(1) = nx
    r(2) = tgrid(it)
    do ip = 1, nnp
      r(3) = pgrid(ip)
      bvec = trilinear(r, bgrid)
      line(ip) = bvec(1)
    enddo

    diffs = line(2:nnp)*line(1:nnp-1)

    do ip = 1, nnp-1
      if (diffs(ip) < 0) then ! change of sign
        r(3) = line(ip)/(line(ip) - line(ip+1))*(pgrid(ip+1) - pgrid(ip)) + pgrid(ip)
        call add_vector(points, r)
      endif
    enddo

    deallocate(line, diffs)

  enddo

  do ip = 1, nnp

    allocate(line(nnt))

    r(1) = nx
    r(3) = pgrid(ip)

    do it = 1, nnt
      r(2) = tgrid(it)
      bvec = trilinear(r, bgrid)
      line(it) = bvec(1)
    enddo

    diffs = line(2:nnt)*line(1:nnt-1)

    do it = 1, nnt-1
      if (diffs(it) < 0) then ! change of sign
        r(2) = line(it)/(line(it) - line(it+1))*(tgrid(it+1) - tgrid(it)) + tgrid(it)
        call add_vector(points, r)
      endif
    enddo

    deallocate(line, diffs)

  enddo

  ! sort points
  ds = 0.05_np ! need to think about this
  ! allocate(pordered(3, 0))
  pordered = points
  pordered = 0
  allocate(npoints(1))
  npoints(1) = 0
  ihcs = 0

  do while (size(points, 2) > 0)
    allocate(ind_line(3, 1))
    ind_line(:, 1) = points(:, 1)
    call remove_vector(points, 1)
    do dir = 1, 2
      do while (size(points, 2) > 0)
        allocate(dists(size(points, 2)))
        do jp = 1, size(points, 2)
          dists(jp) = sum((points(:, jp) - ind_line(:, size(ind_line, 2)))**2)
        enddo
        imin = minloc(dists, 1)
        mindist = dists(imin)
        deallocate(dists)
        if (mindist < ds) then
          call add_vector(ind_line, points(:, imin))
          call remove_vector(points, imin)
        else
          exit
        endif
      enddo
      ind_line = ind_line(:, size(ind_line, 2):1:-1)
    enddo
    pordered(:, count(pordered(1, :) > 0)+1:count(pordered(1, :) > 0)+size(ind_line, 2)) = ind_line
    deallocate(ind_line)
    call add_element(npoints, count(pordered(1, :) > 0))
  enddo

  ! need to substract a tiny amount of the r coord
  pordered(1, :) = pordered(1, :) - 1e-6_np

  open(unit=80, file='output/hcs_start.dat', access='stream', status='replace')
    write(80) size(pordered, 2), gtr(pordered, x, y, z), size(npoints, 1), npoints
  close(80)

  write(10) ringsmax+1, nskip, bytesize, stepsize, int((ubound(npoints, 1) - 1)*2, int32)

  print*, 'There are ', ubound(npoints, 1) - 1, 'components of the hcs'

  tempfile = trim(fileout)//'-hcs-rings.temp'

  nseps1 = 0
  write(40) nseps1

  !$OMP PARALLEL private(iring, iline, dir, ihcs)

  do ihcs = 1, ubound(npoints, 1)-1
    !$OMP SINGLE
    print*, ''
    print*, 'Component ', ihcs
    !$OMP END SINGLE
    do dir = 1, -1, -2 ! do each direction

      !$OMP SINGLE
      if (dir == 1) then
        print*, 'Tracing forwards'
      else
        print*, 'Tracing backwards'
      endif

      open(unit=90, file=tempfile, status='replace', access='stream')
      
      ! get number of start points
      line1 = pordered(:, npoints(ihcs)+1:npoints(ihcs+1))
      line2 = line1
      nlines = size(line1, 2)

      allocate(break(nlines))

      out = .false.

      break = 0
      break(nlines) = 1
      nperring = 0
      nperring(0) = nlines
      slowdown = 1.0_np
      terror = 0

      write_ring = gtr(line1, x, y, z)
      write(20) real(write_ring, bytesize)
      if (assoc_output) write(25) [(iline,iline=1,nlines)]
      write(30) break
      write(90) [(iline,iline=1,nlines)], write_ring
      deallocate(write_ring)
      nseps = 0

      exitcondition = .false.
      !$OMP END SINGLE

      do iring = 1, ringsmax ! loop over number of rings we want

        !$OMP SINGLE

        allocate(endpoints(nlines), association(nlines))

        h0 = stepsize/slowdown
#if debug
      print*, iring, nlines
#endif
        !$OMP END SINGLE

        !$OMP WORKSHARE
        endpoints = 0
        !$OMP END WORKSHARE

        !$OMP DO private(r, h, out)
        do iline = 1, nlines ! loop over all points in ring (in parallel do)

          r = line1(:, iline)
          h = h0

          call trace_line(r, dir, h) ! trace line by a distance of h

          line2(:,iline) = line2(:,iline) + (r - line1(:,iline))
          line1(:,iline) = r

          ! counter to see how many points on ring have reached outer boundary
          if (outedge(r)) then
            endpoints(iline) = 1
          else
            endpoints(iline) = 0
          endif

          association(iline) = iline

        enddo
        !$OMP END DO

        !$OMP SINGLE
        maxdist = 0.5_np*h0*slowdown
        nulldist = 2*h0*slowdown
        mindist = maxdist/3
        slowdist = 2.0*nulldist
        !$OMP END SINGLE

        ! remove points from ring if necessary
        call remove_points(nlines)

        !$OMP SINGLE
        allocate(nearnull(nlines))
        slowdown = 1.0_np
        !$OMP END SINGLE

        !$OMP WORKSHARE
        nearnull = 0
        !$OMP END WORKSHARE

        !$OMP DO private(jnull)
        do iline = 1, nlines
          do jnull = 1, nnulls
            if (signs(jnull) == dir) cycle
            if (dist(rnulls(:,jnull), line1(:, iline)) < slowdist .or. &
              dist(rnullsalt(:,jnull), line1(:, iline)) < slowdist) then
              nearnull(iline) = 1
              exit
            endif
          enddo
        enddo
        !$OMP END DO

        !$OMP SINGLE
        nearflag = sum(nearnull)
        if (nearflag > 0) then
          ! print*, 'slowing down, adding points near null'
          slowdown = 2.0_np
        endif
        !$OMP END SINGLE

        ! add points to ring if necessary
        call add_points(nlines)

        !$OMP SINGLE
        do iline = 1, nlines
          call edgecheck(line1(:, iline))
        enddo
        !$OMP END SINGLE

        !$OMP SINGLE
        deallocate(nearnull)
        !$OMP END SINGLE

        ! determine if any lines are separators
        if (nearflag > 0) call sep_detect(nlines, 0, iring, dir)

        !$OMP SINGLE

        if (nlines > pointsmax) then
        ! exit if too many points on ring
          print*, 'Too many points on ring', nlines, iring
          exitcondition = .true.
        endif

        if (nlines == 0) then
          ! exit if all points have reached outer boundary (left box)
          print*, 'All fan points have reached the outer boundary', iring
          exitcondition = .true.
        endif

        if (terror == 1) then
          print*, 'Tracing has failed', iring
          exitcondition = .true.
        endif

        if (iring - samemax - 1 > 0) then
          if (all(nperring(iring - samemax - 1:iring-1) == nperring(iring-1))) then
            print*, 'Ring seems to have stopped growing', iring
            print*, "Maybe there's a null that hasn't been found"
            exitcondition = .true.
          endif
        endif

        if (exitcondition) deallocate(endpoints, association)
        nrings = iring
        !$OMP END SINGLE

        if (exitcondition) exit

        !$OMP SINGLE
        nperring(iring) = nlines
        write_ring = gtr(line1, x, y, z)
        if (modulo(iring, nskip) == 0) then
          if (bytesize == real64) then
            write(20) write_ring
          else
            write(20) real(write_ring, bytesize)
          endif
          write(30) break
        endif
        write(90) association, write_ring

        deallocate(endpoints, association, write_ring)
        !$OMP END SINGLE

      enddo

      !$OMP SINGLE
      write(10) nperring(::nskip)

      ! trace separators...
      print*, "Tracing separators"
      write(40, pos=uptonullconn)
      if (nseps > 0) then
        do isep = 1, nseps
          read(40) nullnum1, nullnum2, ringnum, linenum
          allocate(rsep(3, ringnum+2))
          do iring = ringnum, 0, -1
            call file_position(iring, nperring, linenum, ia, ip)
            read(90, pos=ia) linenum
            read(90, pos=ip) rsep(:,iring+1)
          enddo
          rsep(:,ringnum+2) = rnullsreal(:,nullnum2)
          write(50) ringnum+2
          write(50) rsep
          deallocate(rsep)
        enddo
      endif
      nseps1 = nseps1 + nseps
      uptonullconn = uptonullconn + nseps*16_int64

      if (assoc_output) then
        do iring = nskip, nskip*(nrings/nskip), nskip
          allocate(association(nperring(iring)))
          call file_position(iring, nperring, 1, ia, ip)
          read(90, pos=ia) association
          do iskip = 1, nskip-1
            allocate(assoc_temp(nperring(iring-iskip)))
            call file_position(iring-iskip, nperring, 1, ia, ip)
            read(90, pos=ia) assoc_temp
            do iline = 1, nperring(iring)
              association(iline) = assoc_temp(association(iline))
            enddo
            deallocate(assoc_temp)
          enddo
          write(25) association
          deallocate(association)
        enddo
      endif

      if (nrings == ringsmax) print*, "Reached maximum number of rings"

      print*, 'number of separators=', nseps, 'number of rings', nrings

      deallocate(line1, line2, break)

      close(90, status='delete')
      !$OMP END SINGLE

    enddo
  enddo

  !$OMP END PARALLEL

  close(10)
  close(20)
  write(40, pos=1) nseps1
  close(40)
  close(50)

  call system_clock(tstop, count_rate)
  print*, 'TIME = ', dble(tstop - tstart)/dble(count_rate), "(", dble(tstop - tstart)/dble(count_rate)/60, "minutes)"
  print*, 'Done!'

end program