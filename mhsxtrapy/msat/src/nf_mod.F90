! module for nullfinder
module nf_mod
  
  use params

  implicit none

  contains

  subroutine bilin_test(cbx,cby,cbz,inull)
    ! tests surface of cube using bilinear method
    implicit none

    integer(int32) :: i, inull

    integer(int32) :: cross, sign
    integer(int32) :: edge

    real(np), dimension(2, 2, 2) :: cbx,cby,cbz

    real(np), dimension(2, 2) :: facex, facey, facez

    integer(int32), dimension(6,6) :: test
    integer(int32), dimension(6) :: test2
    integer(int32), dimension(3) :: test3
    
    ! test is an array where the first index refers to the face number (1:6) and the second index is follows:
    ! 1: x-y intersection (1/0)
    ! 2: sign of z at this intersection (-1/1 /100 if null on face)
    ! 3: xz intersection (1/0)
    ! 4: sign of y on this intersection (-1/1 /100 if null on face)
    ! 5: yz intersection (1/0)
    ! 6: sign on x on this inersection (-1/1 /100 if null on face)

    inull = 0 ! variable defining null. 1=null, 0=no null
    test = 0 ! set test array to zero
    edge = 0 ! variable defining null on edge. 1=null, 0=no null

    !face #1 : (bottom face)
    facex = cbx(:,:,1)
    facey = cby(:,:,1)
    facez = cbz(:,:,1)

    call face_solve(facex, facey, facez, cross, sign) !check if zeroes of x and y components of B cross on face
    test(1,1) = cross !do they cross (1/0)
    test(1,2) = sign !sign (-1/1)
    call face_solve(facey, facez, facex, cross, sign) !check if zeroes of y and z components of B cross on face
    test(1,3) = cross !do they cross (1/0)
    test(1,4) = sign !sign (-1/1)
    call face_solve(facex, facez, facey, cross, sign) !check if zeroes of x and z components of B cross on face
    test(1,5) = cross !do they cross (1/0)
    test(1,6) = sign !sign (-1/1)

    edge = edge + edge_check(facex, facey, facez) !check for null on edge (with separator lying along edge)

    ! face #2 : left face
    facex = cbx(1,:,:)
    facey = cby(1,:,:)
    facez = cbz(1,:,:)

    call face_solve(facex, facey, facez, cross, sign)
    test(2,1) = cross
    test(2,2) = sign
    call face_solve(facey, facez, facex, cross, sign)
    test(2,3) = cross
    test(2,4) = sign
    call face_solve(facex, facez, facey, cross, sign)
    test(2,5) = cross
    test(2,6) = sign

    edge = edge + edge_check(facex, facey, facez)

    ! face #3 : right face
    facex = cbx(2,:,:)
    facey = cby(2,:,:)
    facez = cbz(2,:,:)

    call face_solve(facex, facey, facez, cross, sign)
    test(3,1) = cross
    test(3,2) = sign
    call face_solve(facey, facez, facex, cross, sign)
    test(3,3) = cross
    test(3,4) = sign
    call face_solve(facex, facez, facey, cross, sign)
    test(3,5) = cross
    test(3,6) = sign

    edge = edge + edge_check(facex, facey, facez)

    ! face 4: front face
    facex = cbx(:,1,:)
    facey = cby(:,1,:)
    facez = cbz(:,1,:)

    call face_solve(facex, facey, facez, cross, sign)
    test(4,1) = cross
    test(4,2) = sign
    call face_solve(facey, facez, facex, cross, sign)
    test(4,3) = cross
    test(4,4) = sign
    call face_solve(facex, facez, facey, cross, sign)
    test(4,5) = cross
    test(4,6) = sign

    edge = edge + edge_check(facex, facey, facez)

    ! face 5: rear face
    facex = cbx(:,2,:)
    facey = cby(:,2,:)
    facez = cbz(:,2,:)

    call face_solve(facex, facey, facez, cross, sign)
    test(5,1) = cross
    test(5,2) = sign
    call face_solve(facey, facez, facex, cross, sign)
    test(5,3) = cross
    test(5,4) = sign
    call face_solve(facex, facez, facey, cross, sign)
    test(5,5) = cross
    test(5,6) = sign

    edge = edge + edge_check(facex, facey, facez)

    ! face 6: top face
    facex = cbx(:,:,2)
    facey = cby(:,:,2)
    facez = cbz(:,:,2)

    call face_solve(facex, facey, facez, cross, sign)
    test(6,1) = cross
    test(6,2) = sign
    call face_solve(facey, facez, facex, cross, sign)
    test(6,3) = cross
    test(6,4) = sign
    call face_solve(facex, facez, facey, cross, sign)
    test(6,5) = cross
    test(6,6) = sign

    edge = edge + edge_check(facex, facey, facez)

    ! check for nulls within cell

    ! total of each column in 'test'
    test2 = sum(test,1)

    test3 = 0
    do i = 1, 3
      if (test2(2*i-1) >= 0) then ! if there is a crossing
        if (abs(test2(2*i)) < test2(2*i-1)) test3(i) = 1
      endif
    enddo

    ! check for null in the cell
    if (sum(test3) == 3) inull = 1

    ! check for null on surface of cell
    if (sum(test(:,2)) > 10 .or. sum(test(:,4)) > 10 .or. sum(test(:,6)) > 10) inull = 1

    ! check for null on edge with spine/separator directed along that edge
    if (edge > 0) inull = 1

  end

  !********************************************************************************

  logical function check(x,y)
    !checks of x and y are between 0 and 1
    real(np) :: x, y

    check = (x >= 0.0_np) .and. (x <= 1.0_np) .and. (y >= 0.0_np) .and. (y <= 1.0_np)
  end

  !********************************************************************************

  subroutine face_solve(facex,facey,facez,cross,sign)
    ! tests if the zeroes on facex and facey cross, and if so, 
    ! what the sign of facez is at the crossing point.
    ! Uses the Method described in Haynes et al. 2007

    implicit none
    real(np), dimension(2, 2) :: facex, facey, facez
    integer(int32) :: cross, sign, nsol
    real(np) :: a1, b1, c1, d1 ! bilinear coefficients (facex)
    real(np) :: a2, b2, c2, d2 ! bilinear coefficients (facey)
    real(np) :: a, b, c ! quadratic coefficients ax^2+bx+c=0

    real(np), dimension(2,2) :: x, y
    real(np) :: det

    real(np) :: zcomp

    logical :: same_sign, not_zero

    x = -1
    y = -1

    sign = 0
    cross = 0
    nsol = 0

    same_sign = count(facex > zero) == 4 .or. &
                count(facex < -zero) == 4 .or. &
                count(facey > zero) == 4 .or. &
                count(facey < -zero) == 4

    not_zero = sum(abs(facex) + abs(facey) + abs(facez)) > 12*zero

    ! if there could be an intersection of the zeroes on facex and face y, or if the faces are not completely zero...
    if (.not. same_sign .and. not_zero) then
      ! get bilinear coefficients
      a1 = facex(1,1)
      b1 = facex(2,1) - facex(1,1)
      c1 = facex(1,2) - facex(1,1)
      d1 = facex(2,2) - facex(2,1) - facex(1,2) + facex(1,1)

      a2 = facey(1,1)
      b2 = facey(2,1) - facey(1,1)
      c2 = facey(1,2) - facey(1,1)
      d2 = facey(2,2) - facey(2,1) - facey(1,2) + facey(1,1)

      ! get quadratic coefficients (ax^2 + bx + c = 0)
      a = b1*d2 - b2*d1
      b = (a1*d2 - a2*d1) + (b1*c2 - c1*b2)
      c = a1*c2 - a2*c1

      ! determinant of quadratic
      det = b**2 - 4*a*c

      if (det >= 0) then ! there is a solution
        if (abs(a) < zero) then !have to solve linear
          if (b /= 0.0_np) then
            x(1,:) = -c/b ! solution exists
            ! x(1,2) = -c/b ! solution exists
            nsol = 1
          endif
        else ! have to solve quadratic
          if (det == 0.0_np) then ! one solution
            x(1,:) = -b/(2*a)
            ! x(1,2) = -b/(2*a)
            nsol = 1
          else ! two solutions
            x(1,:) = (-b + sqrt(det))/(2*a)
            ! x(1,2) = (-b + sqrt(det))/(2*a)
            x(2,:) = (-b - sqrt(det))/(2*a)
            ! x(2,2) = (-b - sqrt(det))/(2*a)
            nsol = 2
          endif
        endif
      endif
      if (nsol > 0) then
        y(1,1) = -(a1 + b1*x(1,1))/(c1 + d1*x(1,1))
        y(1,2) = -(a2 + b2*x(1,2))/(c2 + d2*x(1,2))
        if (nsol == 2) then
          y(2,1) = -(a1 + b1*x(2,1))/(c1 + d1*x(2,1))
          y(2,2) = -(a2 + b2*x(2,2))/(c2 + d2*x(2,2))
        endif
      else
        ! if the x equation isn't solvable then try the y equation
        ! get quadratic coefficients (ay^2 + by + c = 0)
        a = c1*d2 - c2*d1
        b = (a1*d2 - a2*d1) - (b1*c2 - c1*b2)
        c = a1*b2 - a2*b1

        det = b**2 - 4*a*c

        if (det >= 0) then ! there is a solution
          if (abs(a) < zero) then ! have to solve linear
            if (b /= 0.0_np) then
              y(1,:) = -c/b ! solution exists
              nsol = 1
            endif
          else ! have to solve quadratic
            if (det == 0.0_np) then !one solution
              y(1,:) = -b/(2*a)
              nsol = 1
            else ! two solutions
              y(1,:) = (-b + sqrt(det))/(2*a)
              y(2,:) = (-b - sqrt(det))/(2*a)
              nsol = 2
            endif
          endif
          if (nsol > 0) then
            x(1,1) = -(a1 + c1*y(1,1))/(b1 + d1*y(1,1))
            x(1,2) = -(a2 + c2*y(1,2))/(b2 + d2*y(1,2))
            if (nsol == 2) then
              x(2,1) = -(a1 + c1*y(2,1))/(b1 + d1*y(2,1))
              x(2,2) = -(a2 + c2*y(2,2))/(b2 + d2*y(2,2))
            endif
          endif
        endif
      endif

      if (check(x(1,1), y(1,1))) then ! if x and y are on face (between 0 and 1)
        cross = cross + 1

        zcomp = bilinear_cell(x(1,1), y(1,1), facez) ! 'z' component of field at crossing point
        if (zcomp > zero) then
          sign = sign + 1
        else if (zcomp < -zero) then
          sign = sign - 1
        else
          sign = sign + 100
        endif
      endif

      if (check(x(1,2), y(1,2))) then
        cross = cross + 1

        zcomp = bilinear_cell(x(1,2), y(1,2), facez)
        if (zcomp > zero) then
          sign = sign + 1
        else if (zcomp < -zero) then
          sign = sign - 1
        else
          sign = sign + 100
        endif
      endif

      if (check(x(2,1), y(2,1))) then
        cross = cross + 1

        zcomp = bilinear_cell(x(2,1), y(2,1), facez)
        if (zcomp > zero) then
          sign = sign + 1
        else if (zcomp < -zero) then
          sign = sign - 1
        else
          sign = sign + 100
        endif
      endif
    
      if (check(x(2,2), y(2,2))) then
        cross = cross + 1

        zcomp = bilinear_cell(x(2,2), y(2,2), facez)
        if (zcomp > zero) then
          sign = sign + 1
        else if (zcomp < -zero) then
          sign = sign - 1
        else
          sign = sign + 100
        endif
      endif

    endif

  end subroutine

  !********************************************************************************

  function edge_check(facex,facey,facez)
    ! check for null along edges of a cell face
    implicit none
    real(np), dimension(2, 2) :: facex,facey,facez
    real(np), dimension(2) :: linex, liney , linez
    integer(int32) :: edge_check

    edge_check = 0

    linex = facex(:, 1)
    liney = facey(:, 1)
    linez = facez(:, 1)

    edge_check = edge_check + line_check(linex, liney, linez) ! check for null along this edge

    linex = facex(:, 2)
    liney = facey(:, 2)
    linez = facez(:, 2)

    edge_check = edge_check + line_check(linex, liney, linez)

    linex = facex(1, :)
    liney = facey(1, :)
    linez = facez(1, :)

    edge_check = edge_check + line_check(linex, liney, linez)

    linex = facex(2, :)
    liney = facey(2, :)
    linez = facez(2, :)

    edge_check = edge_check + line_check(linex, liney, linez)

    if (edge_check > 0) edge_check = 1 !if nulls are found, set the counter to 1

  end

  !********************************************************************************

  function line_check(linex, liney, linez)
    !checks if null lies on an edge

    implicit none

    real(np), dimension(2) :: linex, liney, linez
    integer(int32) :: line_check

    if ((linex(1) == 0.0_np .and. linex(2) == 0.0_np) .and. &
      (liney(1) == 0.0_np .and. liney(2) == 0.0_np)) then
      if (linez(1)*linez(2) <= 0.0_np) line_check = 1
    else if ((linex(1) == 0.0_np .and. linex(2) == 0.0_np) .and. &
      (linez(1) == 0.0_np .and. linez(2) == 0.0_np)) then
      if (liney(1)*liney(2) <= 0.0_np) line_check = 1
    else if ((linez(1) == 0.0_np .and. linez(2) == 0.0_np) .and. &
      (liney(1) == 0.0_np .and. liney(2) == 0.0_np)) then
      if (linex(1)*linex(2) <= 0.0_np) line_check = 1
    else
      line_check = 0
    endif

  end

  !********************************************************************************

  function bilinear_cell(x, y, square)
    ! interpolate the value of a function at (x, y) from the 4 corner values

    implicit none
    
    real(np) :: x, y, square(2, 2)
    real(np) :: y1, y2, line(2)
    real(np) :: bilinear_cell

    line = (1 - x)*square(1, :) + x*square(2, :)

    bilinear_cell = line(1)*(1 - y) + line(2)*y
  end

  !********************************************************************************

  function allsame(cube)
    !determines whether all the vertices of the cube have the same sign or not
    implicit none
    real(np) :: cube(2,2,2)
    logical :: allsame

    if (count(cube > zero) == 8 .or. count(cube < -zero) == 8) then
      allsame = .true.
    else
      allsame = .false.
    endif

  end function

end module
