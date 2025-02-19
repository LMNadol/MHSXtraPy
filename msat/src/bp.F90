!code to find the separatrix surface and separators of null points
program hcs

  use omp_lib
  use params
  use common
  use trace
  use ring

  implicit none

  integer(int64) :: tstart, tstop, count_rate !to time program

  real(np) :: r(3)
  real(np) :: h, h0
  real(np) :: slowdown
  integer(int32) :: nearflag

  integer(int32) :: iring, iline, inull, jnull, dir

  integer(int32) :: nlines
  integer(int32) :: nrings
  integer(int32) :: nperring(0:ringsmax)

  logical :: exitcondition, out

  integer(int32) :: ihcs, it, nnt, nnp, imin, jp, nsplit, iseg
  integer(int32), dimension(:), allocatable :: npoints, breakbp
  real(np), dimension(:), allocatable :: tgrid, pgrid, line, diffs, dists
  real(np), dimension(:,:), allocatable :: points, pordered, baldpatch
  real(np), dimension(3) :: lvec, bvec, vec, field1, field2, perp, midpoint
  real(np) :: ds

  ! file writing
  integer(int64) :: ia, ib, ip
  integer(int32) :: isep, nullnum1, nullnum2, linenum, ringnum
  real(np), dimension(:,:), allocatable :: rsep, rring, linewrite
  integer(int32), dimension(:), allocatable :: brk

  print*,'#######################################################################'
  print*,'#                      Separatrix Surface Finder                      #'
  print*,'#######################################################################'

  ! call omp_set_num_threads(nproc) ! have it work on 4 threads (If machine has >4 cores this should be larger, if fewer than 4 coures, this should be smaller)

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

  ! stores all info regarding size of rings
  open(unit=10,file=trim(fileout)//'-ringinfo-bp.dat',status='replace',access='stream')
  ! stores all the coordinates of rings in original coordinate system
  open(unit=20,file=trim(fileout)//'-rings-bp.dat',status='replace',access='stream')
  ! stores all the connection info about each separator
  open(unit=40,file=trim(fileout)//'-connectivity-bp.dat',status='replace',access='stream')
  ! stores all the coordinates of the separator lines in original coordinate system
  open(unit=50,file=trim(fileout)//'-separators-bp.dat',status='replace',access='stream')

  ! calculate the points along the hcs line
  nsplit = 10
  nnt = nsplit*(ny-1)+1
  nnp = nsplit*(nz-1)+1

  tgrid = real([(it,it=0,nnt-1)], np)/nsplit + 1
  pgrid = real([(ip,ip=0,nnp-1)], np)/nsplit + 1
  ! has issues right on the edge...
  tgrid(1) = tgrid(1) + 1e-6_np
  tgrid(nnt) = tgrid(nnt) - 1e-6_np
  allocate(points(3, 0))

  do it = 1, nnt

    allocate(line(nnp))

    r(1) = 1
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

  print*, 'done theta'

  do ip = 1, nnp

    allocate(line(nnt))

    r(1) = 1
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

  print*, 'done phi'
  print*, size(points, 2)

  ! sort points
  ds = maxval([maxval(y(2:ny) - y(1:ny-1)), maxval(z(2:nz) - z(1:nz-1))])
  allocate(pordered(3, 0))
  allocate(npoints(1))
  npoints(1) = 0
  do while (size(points, 2) > 0)
    call add_vector(pordered, points(:, 1))
    call remove_vector(points, 1)
    ! add points in the "forward" direction
    ip = 1
    do while (size(points, 2) > 0)
      allocate(dists(size(points, 2)))
      do jp = 1, size(points, 2)
        dists(jp) = sum((points(:,jp) - pordered(:, ip))**2)
      enddo
      imin = minloc(dists, 1)
      mindist = dists(imin)
      deallocate(dists)
      if (mindist < ds) then
        call add_vector(pordered, points(:, imin))
        call remove_vector(points, imin)
      else
        exit
      endif
      ip = ip + 1
    enddo
    ! add points in the "backward" direction
    do while (size(points, 2) > 0)
      allocate(dists(size(points, 2)))
      do jp = 1, size(points, 2)
        dists(jp) = sum((points(:, jp) - pordered(:, 1))**2)
      enddo
      imin = minloc(dists, 1)
      mindist = dists(imin)
      deallocate(dists)
      if (mindist < ds) then
        call add_vector(pordered, points(:, imin), 1)
        call remove_vector(points, imin)
      else
        exit
      endif
    enddo
    call add_element(npoints, size(pordered, 2))
  enddo

  ! need to add a tiny amount of the r coord
  ! pordered(1, :) = pordered(1, :) + 1e-6_np

  print*, 'done ordering'

  allocate(baldpatch(3,0), breakbp(0))
  do ip = 1, size(npoints, 1)-1
    ! print*, ip
    do iseg = npoints(ip)+1, npoints(ip+1)-1
      lvec = pordered(:, iseg+1) - pordered(:, iseg)
      midpoint = (pordered(:, iseg+1) + pordered(:, iseg))/2
      perp = [0.0_np, lvec(3), -lvec(2)]
      ! print*, lvec
      ! print*, midpoint, shape(bgrid)
      ! print*, perp
      bvec = trilinear(midpoint, bgrid)
      field1 = trilinear(midpoint+perp*1e-4_np, bgrid)
      field2 = trilinear(midpoint-perp*1e-4_np, bgrid)
      if (field1(1) > field2(1)) then
        perp = -perp
      endif
      if (dot(bvec, perp) > 0) then ! we have a bald patch
        ! print*, iseg
        ! print*, size(baldpatch, 2), pordered(:, iseg), baldpatch(:, size(baldpatch, 2))
        if (size(baldpatch, 2) /= 0) then
          if (.not. all(pordered(:, iseg) == baldpatch(:, size(baldpatch, 2)))) then
              breakbp(size(baldpatch, 2)) = 1
              call add_vector(baldpatch, pordered(:, iseg))
              call add_element(breakbp, 0)
            endif
          endif
          call add_vector(baldpatch, pordered(:, iseg+1))
          call add_element(breakbp, 0)
        else
          call add_vector(baldpatch, pordered(:, iseg))
          call add_element(breakbp, 0)
          call add_vector(baldpatch, pordered(:, iseg+1))
          call add_element(breakbp, 0)
      endif
    enddo
  enddo

  print*, 'done', size(baldpatch, 2)
  open(unit=9, file='test.dat', status='replace', access='stream')
  write(9) size(baldpatch, 2), gtr(baldpatch, x, y, z)
  close(9)

  write(10) ringsmax+1

  ihcs = 1

  !$OMP PARALLEL private(iring, iline, inull)
  do dir = 1, -1, -2 ! do each direction
    !$OMP SINGLE
    ! get number of start points
    line1 = pordered(:, npoints(ihcs)+1:npoints(ihcs+1))
    line2 = line1
    nlines = size(line1, 2)

    allocate(break(nlines))

    out = .false.

    open(unit=90, access='stream', status='scratch')
    open(unit=95, access='stream', status='scratch')

    break = 0
    break(nlines) = 1
    nseps = 0
    nperring = 0
    nperring(0) = nlines
    slowdown = 1.0_np
    terror = 0

    linewrite = gtr(line1, x, y, z)
    write(90) [(iline,iline=1,nlines)], break, linewrite
    write(20) [(iline,iline=1,nlines)], break, linewrite
    deallocate(linewrite)

    exitcondition = .false.
    !$OMP END SINGLE

    do iring = 1, ringsmax ! loop over number of rings we want

      !$OMP SINGLE

      allocate(endpoints(nlines), association(nlines))

      h0 = 1e-1_np/slowdown

      !$OMP END SINGLE

      !$OMP WORKSHARE
      endpoints = 0
      !$OMP END WORKSHARE

      !$OMP DO private(r, h, out)
      do iline = 1, nlines ! loop over all points in ring (in parallel do)

        r = line1(:,iline)
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
      maxdist = h0*slowdown
      nulldist = 2*h0*slowdown
      mindist = maxdist/3
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
          if (dist(rnulls(:,jnull), line1(:, iline)) < 2.5*nulldist .or. &
            dist(rnullsalt(:,jnull), line1(:, iline)) < 2.5*nulldist) then
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
        call edgecheck(line1(:,iline))
      enddo
      !$OMP END SINGLE

      !$OMP SINGLE
      deallocate(nearnull)
      !$OMP END SINGLE

      ! determine if any lines are separators
      if (nearflag > 0) call sep_detect(nlines,inull,iring,dir)

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

      if (exitcondition) deallocate(endpoints, association)
      !$OMP END SINGLE

      if (exitcondition) then
        nrings = iring
        exit
      endif

      !$OMP SINGLE
      nperring(iring) = nlines
      ! Write ring and data to file separator????.dat
      linewrite = gtr(line1, x, y, z)
      write(90) association, break, linewrite
      write(20) association, break, linewrite

      deallocate(endpoints, association, linewrite)
      !$OMP END SINGLE

    enddo

    !$OMP SINGLE
    write(10) nperring

    ! trace separators...
    write(95, pos=1)
    if (nseps > 0) then
      print*, "Tracing separators"
      do isep = 1, nseps
        read(95) nullnum1, nullnum2, ringnum, linenum
        allocate(rsep(3, ringnum+2))
        do iring = ringnum, 0, -1
          call file_position(iring, nperring, linenum, ia, ib, ip)
          read(90, pos=ia) linenum
          read(90, pos=ip) rsep(:,iring+1)
        enddo
        rsep(:,ringnum+2) = rnullsreal(:,nullnum2)
        write(50) ringnum+2
        write(50) rsep
        deallocate(rsep)
      enddo
    endif

    ! don't need scratch ring file anymore
    close(95)
    close(90)

    if (nrings == ringsmax) print*, "Reached maximum number of rings"

    print*, 'number of separators=', nseps, 'number of rings', nrings

    deallocate(line1, line2, break)
    !$OMP END SINGLE

  enddo

  !$OMP END PARALLEL

  close(10)
  close(20)
  write(40) -1
  close(40)
  close(50)

  call system_clock(tstop, count_rate)
  print*, 'TIME = ', dble(tstop - tstart)/dble(count_rate), "(", dble(tstop - tstart)/dble(count_rate)/60, "minutes)"
  print*, 'Done!'

end program