module ring
! this module contains parameters involved in tracing rings, and subroutines to add points or remove points from rings. Also included is the subroutine (at_null) which determines if a point on the ring has reached a null.

use params
use common
use trace

integer(int32) :: nseps

real(np) :: maxdist, mindist, nulldist

real(np), dimension(:,:), allocatable :: line1, line2, add1, add2
integer(int32), dimension(:), allocatable :: break, association, remove, endpoints, nearnull

contains

  subroutine add_points(nlines)
    ! adds points to rings if required
    implicit none

    integer(int32) :: nlines
    integer(int32) :: iline, nxtline, iadd, nadd
    real(np) :: maxdist0

    logical, dimension(nlines) :: add_true
    real(np), dimension(:, :), allocatable :: line1_temp, line2_temp
    integer, dimension(:), allocatable :: break_temp, association_temp

    !$OMP SINGLE
    allocate(add1(3,nlines), add2(3,nlines))
    !$OMP END SINGLE

    !$OMP WORKSHARE
    add1 = 0.0_np
    add2 = 0.0_np
    !$OMP END WORKSHARE

    ! test for gaps. Where gaps need to be filled, put this info into add1, add2
    !$OMP DO private(iline, maxdist0, nxtline)
    do iline = 1, nlines !loop over each point in ring
      maxdist0 = maxdist
      if (break(iline) == 0) then
        if (iline < nlines) then !if not the last point
          nxtline = iline+1
        else
          nxtline = 1
        endif
        if (nearnull(iline) == 1 .or. nearnull(nxtline) == 1) maxdist0 = maxdist/2.0_np
        if (dist(line2(:,iline),line2(:,nxtline)) > maxdist0) then !if two adjacent points too far away
          ! add point half way between two points
          add1(:,iline) = line1(:,iline) + 0.5_np*(line2(:,nxtline)-line2(:,iline)) 
          add2(:,iline) = line2(:,iline) + 0.5_np*(line2(:,nxtline)-line2(:,iline))
        endif
      endif
    enddo
    !$OMP END DO

    ! ! add new points
    ! ! where the 'add' array has a point to be added, add this point
    ! nadd = 1
    ! do iline = 1, nlines
    !   if (add1(1,iline) > 1e-1_np) then !if |add(:,iline)| is not zero, all points > (1,1,1)
    !     iadd = iline + nadd
    !     !$OMP SECTIONS
    !     !$OMP SECTION
    !     call add_vector(line1,add1(:,iline),iadd) !add elements to every array
    !     !$OMP SECTION
    !     call add_vector(line2,add2(:,iline),iadd)
    !     !$OMP SECTION
    !     call add_element(break,0,iadd)
    !     !$OMP SECTION
    !     if (iline /= nlines) then
    !       call add_element(association,association(iadd-1),iadd)
    !     else
    !       call add_element(association,1,iadd)
    !     endif
    !     !$OMP END SECTIONS
    !     nadd = nadd + 1
    !   endif
    ! enddo

    ! !$OMP BARRIER
    !$OMP SINGLE
    add_true = add1(1, :) > 1e-1_np
    if (any(add_true)) then
      nadd = count(add_true)
      call move_alloc(line1, line1_temp)
      call move_alloc(line2, line2_temp)
      call move_alloc(break, break_temp)
      call move_alloc(association, association_temp)
      allocate(line1(3, nlines+nadd), line2(3, nlines+nadd), break(nlines+nadd), association(nlines+nadd))
      iadd = 0
      do iline = 1, nlines
        iadd = iadd + 1
        line1(:, iadd) = line1_temp(:, iline)
        line2(:, iadd) = line2_temp(:, iline)
        break(iadd) = break_temp(iline)
        association(iadd) = association_temp(iline)
        if (add_true(iline)) then
          iadd = iadd + 1
          line1(:, iadd) = add1(:, iline)
          line2(:, iadd) = add2(:, iline)
          break(iadd) = 0
          if (iline /= nlines) then
            association(iadd) = association_temp(iline)
          else
            association(iadd) = 1
          endif
        endif
      enddo
      nlines = nlines + nadd
    endif

    deallocate(add1, add2)

    !$OMP END SINGLE

  end subroutine

  !********************************************************************************

  subroutine remove_points(nlines)
    ! removes points from rings as required
    implicit none

    integer(int32) :: nlines
    integer(int32) :: iline, nxtline, iremove, nremove, break1

    logical, dimension(nlines) :: remove_false1
    logical, dimension(3, nlines) :: remove_false2

    !$OMP SINGLE
    allocate(remove(nlines))
    !$OMP END SINGLE

    !$OMP WORKSHARE
    remove = 0
    !$OMP END WORKSHARE

    !$OMP SINGLE
    break1 = break(1)
    do iline = 2, nlines
      if (endpoints(iline) == 1 .or. break(iline) == 2) then ! remove points that have left the simulation or stuck
        if (break(iline) == 2) endpoints(iline) = 1
        remove(iline) = 1
        break(iline-1) = 1
      endif
    enddo
    if (endpoints(1) == 1 .or. break1 == 2) then ! remove points that have left the simulation or stuck
      if (break(1) == 2) endpoints(1) = 1
      remove(1) = 1
      break(nlines) = 1
    endif

    ! check for too tightly spaced points, flag points to be removed
    do iline = 1, nlines ! loop over all points
      if (break(iline) == 0) then
        if (iline /= nlines) then ! if not end point
          nxtline = iline+1
        else
          nxtline = 1
        endif
        if (dist(line2(:,iline), line2(:,nxtline)) < mindist .and. remove(iline) == 0) then ! if iline and iline+1 are too near
          if (break(nxtline) == 1) then
            remove(iline) = 1
          else
            remove(nxtline) = 1
          endif
        endif
      endif
    enddo
    ! !$OMP END SINGLE

    ! ! remove points
    ! nremove = 0
    ! if (nlines > nstart .or. sum(endpoints) /= 0) then ! if the number of points isn't too small...
    !   do iline = 1, nlines
    !     if (nlines <= nstart .and. sum(endpoints(iline:nlines)) == 0) exit
    !     if (remove(iline) == 1) then ! if point is flagged to be removed, then remove
    !       iremove = iline - nremove
    !       !$OMP SECTIONS
    !       !$OMP SECTION
    !       call remove_vector(line1,iremove)
    !       !$OMP SECTION
    !       call remove_vector(line2,iremove)
    !       !$OMP SECTION
    !       call remove_element(association,iremove)
    !       !$OMP SECTION
    !       call remove_element(break,iremove)
    !       !$OMP END SECTIONS
    !       nremove = nremove + 1
    !     endif
    !   enddo
    ! endif
    ! !$OMP BARRIER
    ! !$OMP SINGLE

    if (nlines > nstart .or. sum(endpoints) /= 0) then
      remove_false1 = remove == 0
      remove_false2 = spread(remove_false1, 1, 3)
      nlines = count(remove_false1) ! nlines - count(remove == 1)
      line1 = reshape(pack(line1, remove_false2), [3, nlines])
      line2 = reshape(pack(line2, remove_false2), [3, nlines])
      association = reshape(pack(association, remove_false1), [nlines])
      break = reshape(pack(break, remove_false1), [nlines])
    endif

    ! nlines = size(line1, 2)

    deallocate(remove)

    !$OMP END SINGLE

  end subroutine

  !********************************************************************************

  subroutine sep_detect(nlines, nullnum, nring, sign)
    ! determine whether a point in a ring is near a null
    ! if so, integrate it and its neighbours forward to see if any of them diverge around the null to find a separator

    implicit none

    integer(int32) :: nlines
    integer(int32) :: nullnum ! the null the fan is being drawn from
    integer(int32) :: inull, iline, k
    integer(int32) :: nring
    integer(int32) :: sign

    real(np) :: h, tracedist, checkdist, dist1, dist2

    real(np), allocatable :: r(:,:)
    integer(int32), allocatable :: rmap(:)
    integer(int32) :: near(nlines), notnear(nlines), endgap, gapsize
    integer(int32) :: index, count, count1, nr, nnc, nextra, n1, n2
    integer(int32), allocatable :: signof(:)

    integer(int32), parameter :: maxcount = 1000
    real(np), parameter :: h0 = 5e-2_np*stepsize

    tracedist = 3*nulldist ! trace points around null until tracedist away
    checkdist = tracedist + 2*h0 ! distance to check for separator after being iterated to tracedist away

    !$OMP DO ! private(near, notnear, nnc, k, gapsize, endgap, nr, r, rmap, signof, index, count, h)
    do inull = 1, size(rnulls,2)
      if (signs(inull) /= sign) then ! ignore nulls of the same sign (but not nulls with zero/undetermined sign - just in case)
        near = 0
        notnear = 0

        ! first find all points that lie within nulldist and note their index
        do iline = 1, nlines
          if (dist(line1(:,iline),rnulls(:,inull)) < nulldist &
            .or. dist(line1(:,iline),rnullsalt(:,inull)) < nulldist) near(iline) = iline ! point lies within nulldist
        enddo

        if (maxval(near) > 0) then ! if there are any points that lie within this distance
          ! find the longest consectutive number of points not near to the null (should be more not near another null)
          nnc = 0
          do iline = 1, nlines
            if (near(iline) == 0) then
              nnc = nnc + 1
              notnear(iline) = nnc
            else
              notnear(iline) = 0
              nnc = 0
            endif
          enddo
          if (nnc /= 0) then
            iline = 1
            do while (near(iline) == 0)
              nnc = nnc + 1
              notnear(iline) = nnc
              iline = iline + 1
            enddo
          endif

          endgap = maxloc(notnear,1) ! location of last non-near point
          gapsize = notnear(endgap) ! number of points in biggest gap not near null
          nextra = 10 ! number of points to test either side of test points
          nr = nlines - gapsize + 2*nextra ! total number to test (nlines-gapsize near null)

          if (gapsize == 0 .or. nr >= nlines) then
            n1 = 1
            n2 = nlines
            nr = nlines
          else
            ! select all points not in this longest chain to be tested for change in side of the fan
            n1 = modulo(endgap + 1 - nextra - 1, nlines) + 1
            n2 = modulo(nlines + endgap - gapsize + nextra - 1, nlines) + 1
          endif
          allocate(r(3, nr), rmap(nr))

          if (n1 <= n2) then
            r = line1(:, n1:n2)
            rmap = [(k, k=n1, n2)]
          else
            r(:, 1:nlines-n1+1) = line1(:, n1:nlines)
            rmap(1:nlines-n1+1) = [(k, k=n1,nlines)]
            r(:, nlines-n1+2:nr) = line1(:, 1:n2)
            rmap(nlines-n1+2:nr) = [(k, k=1,n2)]
          endif

          allocate(signof(nr))
          signof = 0

          do index = 1, nr
            ! extrapolate points along fieldlines
            count = 0
            do while (dist(r(:,index),rnulls(:,inull)) < tracedist .and. count < maxcount)
              h = h0
              call trace_line(r(:,index),sign,h)
              call edgecheck(r(:,index))
              count = count+1
            enddo

            if (signs(inull)*sign == -1) then
              ! check which side of the null the points end out on
              if (dot(spines(:,inull),r(:,index)-rnulls(:,inull)) > 0) then
                signof(index) = 1
              else
                signof(index) = -1
              endif

              !if theres a change in sign, theres the separator
              if (index /= 1) then
                if (signof(index-1)*signof(index) == -1 .and. break(rmap(index-1)) /= 1 &
                  .and. dist(rnulls(:,inull), r(:,index)) < checkdist &
                  .and. dist(rnulls(:,inull), r(:,index-1)) < checkdist) then
                  dist1 = dist(rnulls(:,inull), r(:,index-1))
                  dist2 = dist(rnulls(:,inull), r(:,index))
#if debug
                  print*, 'Found a separator', nring, rmap(index-1), nlines, nullnum, inull
#else
                  print*, 'Found a separator to null', inull, 'on ring', nring
#endif
                  
                  break(rmap(index-1)) = 1 ! disassociate points so that new points don't get added between them as they diverge around the null
                  nseps = nseps + 1
                  ! write the point's information to the separator file
                  if (dist1 > dist2) then
                    write(40) nullnum, inull, nring, rmap(index)
                  else
                    write(40) nullnum, inull, nring, rmap(index-1)
                  endif
                  if (one_sep_per_ring) exit
                endif
              endif
            endif
            if (count1 == maxcount) then ! points appear to be stuck at the null
              if (rmap(index) /= 1) then
                break(rmap(index)-1) = 2
              else
                break(nlines) = 2
              endif
            endif
            count1 = count
          enddo
          deallocate(r, signof, rmap)
        endif
      endif
    enddo
    !$OMP END DO

  end subroutine

end module
