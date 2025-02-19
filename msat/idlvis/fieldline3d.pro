common shared_var_fl3d, xmin, xmax, ymin, ymax, zmin, zmax, csystem_fl3d, period_fl3d

function trilinear3d_grid, pt, grid
  compile_opt idl2

  ix = floor(pt[0])
  iy = floor(pt[1])
  iz = floor(pt[2])

  x = pt[0] - ix
  y = pt[1] - iy
  z = pt[2] - iz

  cube = grid[ix:ix+1, iy:iy+1, iz:iz+1, *]
  square = reform((1 - z)*cube[*, *, 0, *] + z*cube[*, *, 1, *])
  line = reform((1 - y)*square[*, 0, *] + y*square[*, 1, *])
  return, reform((1 - x)*line[0, *] + x*line[1, *])

end

function getdr, r, x, y, z
  common shared_var_fl3d
  compile_opt idl2

  ix = floor(r[0])
  iy = floor(r[1])
  iz = floor(r[2])
  
  dx = x[ix + 1] - x[ix]
  dy = y[iy + 1] - y[iy]
  dz = z[iz + 1] - z[iz]
  
  xp = x[ix] + (r[0] - ix)*dx
  yp = y[iy] + (r[1] - iy)*dy

  if csystem_fl3d eq 'spherical' then begin
    dr = [dx, xp*dy, xp*sin(yp)*dz]
  endif else if csystem_fl3d eq 'cartesian' then begin
    dr = [dx, dy, dz]
  endif else if csystem_fl3d eq 'cylindrical' then begin
    dr = [dx, xp*dy, dz]
  endif
  return, dr

end

pro edgecheck, r
  common shared_var_fl3d
  compile_opt idl2

  if period_fl3d.hasvalue(1) then begin
    if csystem_fl3d eq 'spherical' then begin
      if period_fl3d[1] then begin
        if r[1] lt ymin or r[1] gt ymax then begin
          if r[1] lt ymin then r[1] = 2*ymin - r[1]
          if r[1] gt ymax then r[1] = 2*ymax - r[1]
          if r[2] lt (zmax + zmin)/2 then begin
            r[2] = r[2] + (zmax - zmin)/2
          endif else begin
            r[2] = r[2] - (zmax - zmin)/2
          endelse
        endif
      endif
      if period_fl3d[2] then begin
        if r[2] le zmin then r[2] = r[2] + (zmax - zmin)
        if r[2] ge zmax then r[2] = r[2] - (zmax - zmin)
      endif
    endif else if csystem_fl3d eq 'cartesian' then begin
      if period_fl3d[0] then begin
        if r[0] le xmin then r[0] = r[0] + (xmax - xmin)
        if r[0] ge xmax then r[0] = r[0] - (xmax - xmin)
      endif
      if period_fl3d[1] then begin
        if r[1] le ymin then r[1] = r[1] + (ymax - ymin)
        if r[1] ge ymax then r[1] = r[1] - (ymax - ymin)
      endif
      if period_fl3d[2] then begin
        if r[2] le zmin then r[2] = r[2] + (zmax - zmin)
        if r[2] ge zmax then r[2] = r[2] - (zmax - zmin)
      endif
    endif else if csystem_fl3d eq 'cylindrical' then begin
      if period_fl3d[1] then begin
        if r[2] lt ymin then r[1] = r[1] + (ymax - ymin)
        if r[2] gt ymax then r[1] = r[1] - (ymax - ymin)
      endif
    endif
  endif
 
end

function outedge, r
  common shared_var_fl3d
  compile_opt idl2

  if period_fl3d.hasvalue(1) then begin
    if csystem_fl3d eq 'cartesian' then begin
      outedge = 0
      if not period_fl3d[0] then begin
        outedge = outedge or r[0] ge xmax or r[0] le xmin
      endif
      if not period_fl3d[1] then begin
        outedge = outedge or r[1] ge ymax or r[1] le ymin
      endif
      if not period_fl3d[2] then begin
        outedge = outedge or r[2] ge zmax or r[2] le zmin
      endif
    endif else if csystem_fl3d eq 'spherical' then begin
      outedge = r[0] ge xmax or r[0] le xmin
      if not period_fl3d[3] then begin
        outedge = outedge or r[1] ge ymax or r[1] le ymin
      endif
      if not period_fl3d[4] then begin
        outedge = outedge or r[2] ge zmax or r[2] le zmin
      endif
    endif else if csystem_fl3d eq 'cylindrical' then begin
      outedge = r[0] ge xmax or r[0] le xmin or r[2] ge zmax or r[2] le zmin
      if not period_fl3d[4] then begin
        outedge = outedge or r[1] ge ymax or r[1] le ymin
      endif
    endif
  endif else begin
      outedge = r[0] ge xmax or r[0] le xmin or r[1] ge ymax or r[1] le ymin or r[2] ge zmax or r[2] le zmin
  endelse
  
  return, outedge

end

function fieldline3d, startpt, bgrid, x, y, z, h, hmin, hmax, epsilon, $
  maxpoints=maxpoints, t_max=t_max, oneway=oneway, boxedge=boxedge, $
  gridcoord=gridcoord, coordsystem=coordsystem, periodicity=periodicity
  common shared_var_fl3d
  compile_opt idl2

  ; startpt[3]: start point for field line
  ; bgrid[nx, ny, nz, 3]: magnetic field 
  ; x[nx], y[ny], z[nz]: grids of the three grid dimensions on which magnetic field given 

  ; h: initial step length
  ; hmin: minimum step length
  ; hmax: maximum step length
  ; epsilon: tolerance to which we require point on field line known

  ; maxpoints: maximum number of points (including starting point) on a fieldline in one direction (maximum of 2*maxpoints - 1 if traced in both directions)
  ; t_max: change maximum value which h can increase by at each iteration
  ; oneway: only let field line trace in one direction (direction of h)
  ; boxedge: create articifical edges to magnetic field grid (2x3 array)
  ; gridcoord: startpt and boxedge is given in grid coordinates, output also in grid coordinates
  ; coordsystem: coordinate system of grid; 'cartesian', 'cylindrical' or 'spherical'
  ; periodicity - set the periodicity of the grid e.g. 'xy' for periodicity in both x and y direction

  if not keyword_set(coordsystem) then coordsystem = 'cartesian'
  csystem_fl3d = coordsystem

  period_fl3d = bytarr(3)
  if not keyword_set(periodicity) then periodicity = ''
  
  if coordsystem eq 'cartesian' then begin
    period_fl3d[0] = strmatch(periodicity, '*x*')
    period_fl3d[1] = strmatch(periodicity, '*y*')
    period_fl3d[2] = strmatch(periodicity, '*z*')
  endif else if coordsystem eq 'cylindrical' then begin
    period_fl3d[0] = 0b
    period_fl3d[1] = not strmatch(periodicity, '*p*')
    period_fl3d[2] = strmatch(periodicity, '*z*')
  endif else if coordsystem eq 'spherical' then begin
    period_fl3d[0] = 0b
    period_fl3d[1] = not strmatch(periodicity, '*t*')
    period_fl3d[2] = not strmatch(periodicity, '*p*')
  endif

  ; define edges of box
  if keyword_set(boxedge) then begin
    period_fl3d[*] = 0b
    boxedge1 = boxedge
    if not keyword_set(gridcoord) then begin
      foreach dim, list(x, y, z), idim do begin
        for im = 0, 1 do begin
          index = max(where(boxedge[im, idim] ge dim))
          boxedge1[im, idim] = index + (boxedge[im, idim] - dim[index])/(dim[index+1] - dim[index])
        endfor    
      endforeach
    endif
    xmin = max([boxedge1[0, 0], 0])
    xmax = min([boxedge1[1, 0], n_elements(x) - 1])
    ymin = max([boxedge1[0, 1], 0])
    ymax = min([boxedge1[1, 1], n_elements(y) - 1])
    zmin = max([boxedge1[0, 2], 0])
    zmax = min([boxedge1[1, 2], n_elements(z) - 1])
  endif else begin
    xmin = 0
    ymin = 0
    zmin = 0
    xmax = n_elements(x) - 1
    ymax = n_elements(y) - 1
    zmax = n_elements(z) - 1
  endelse

  b2 = 0.25d
  b3 = 3d/32d & c3 = 9d/32d
  b4 = 1932d/2197d & c4 = -7200d/2197d & d4 = 7296d/2197d
  b5 = 439d/216d & c5 = -8d & d5 = 3680d/513d & e5 = -845d/4104d
  b6 = -8d/27d & c6 = 2d & d6 = -3544d/2565d & e6 = 1859d/4104d & f6 = -11d/40d

  ;used to determine y_i+1 from y_i if using rkf45 (4th order)
  n1 = 25d/216d & n3 = 1408d/2565d & n4 = 2197d/4104d & n5 = -1d/5d
    
  ;used to determine y_i+1 from y_i if using rkf54 (5th order)
  nn1 = 16d/135d & nn3 = 6656d/12825d & nn4 = 28561d/56430d & nn5 = -9d/50d & nn6 = 2d/55d

  if not keyword_set(gridcoord) then begin
    ix = max(where(startpt[0] ge x))
    iy = max(where(startpt[1] ge y))
    iz = max(where(startpt[2] ge z))

    r0 = startpt
    r0[0] = ix + (startpt[0] - x[ix])/(x[ix+1] - x[ix])
    r0[1] = iy + (startpt[1] - y[iy])/(y[iy+1] - y[iy])
    r0[2] = iz + (startpt[2] - z[iz])/(z[iz+1] - z[iz])
  endif else r0 = startpt

  x0 = startpt[0]
  y0 = startpt[1]
  z0 = startpt[2]

  if (r0[0] lt xmin or r0[0] gt xmax or r0[1] lt ymin or r0[1] gt ymax or r0[2] lt zmin or r0[2] gt zmax) then begin
    print, "Error: Start point not in range"
    print, "Start point is:", startpt
    if (r0[0] lt xmin or r0[0] gt xmax) then print, startpt[0], " (x) is the issue"
    if (r0[1] lt ymin or r0[1] gt ymax) then print, startpt[1], " (y) is the issue"
    if (r0[2] lt zmin or r0[2] gt zmax) then print, startpt[2], " (z) is the issue"
    print, "Bounds are: ", "xmin = ", x[0], " xmax = ", x[-1]
    print, "Bounds are: ", "ymin = ", y[0], " ymax = ", y[-1]
    print, "Bounds are: ", "zmin = ", z[0], " zmax = ", z[-1]

    message, 'Start point error'
  endif else if not (hmin < h and h < hmax) then begin
    message, "You need to satisfy hmin < h < hmax"
  endif

  if not keyword_set(maxpoints) then maxpoints = 50000
  if not keyword_set(t_max) then t_max = 1.1d

  ; #####################################################################
  ; now for the main rkf45 algorithm

  if keyword_set(oneway) then ih = [h] else ih = [h, -h]

  line = list(r0)

  foreach h, ih do begin

    count = 1L
    out = 0
    bounce = 0
    
    while count lt maxpoints do begin

      r0 = line[-1]
      
      dr = getdr(r0, x, y, z)
      mindist = min(dr)*h
      hvec = mindist/dr

      rt = r0
      b = trilinear3d_grid(rt, bgrid)
      k1 = hvec*b/sqrt(total(b^2))
      rt = r0 + b2*k1

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k2 = hvec*b/sqrt(total(b^2))
      rt = r0 + b3*k1 + c3*k2

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k3 = hvec*b/sqrt(total(b^2))
      rt = r0 + b4*k1 + c4*k2 + d4*k3

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k4 = hvec*b/sqrt(total(b^2))
      rt = r0 + b5*k1 + c5*k2 + d5*k3 + e5*k4

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k5 = hvec*b/sqrt(total(b^2))
      rt = r0 + b6*k1 + c6*k2 + d6*k3 + e6*k4 + f6*k5

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k6 = hvec*b/sqrt(total(b^2))

      ; 4th order estimate
      rtest4 = r0 + n1*k1 + n3*k3 + n4*k4 + n5*k5
      ; 5th order estimate
      rtest5 = r0 + nn1*k1 + nn3*k3 + nn4*k4 + nn5*k5 + nn6*k6

      ; optimum stepsize
      diff = rtest5 - rtest4
      err = sqrt(total(diff^2))
      if err gt 0 then begin
        t = (epsilon*abs(h)/(2*err))^0.25d
        if t gt t_max then t = t_max
      endif else t = t_max

      h = t*h
      if abs(h) lt hmin then h = hmin*signum(h)
      if abs(h) gt hmax then h = hmax*signum(h)

      thvec = t*hvec
      
      rt = r0
      b = trilinear3d_grid(rt, bgrid)
      k1 = thvec*b/sqrt(total(b^2))
      rt = r0 + b2*k1

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k2 = thvec*b/sqrt(total(b^2))
      rt = r0 + b3*k1 + c3*k2

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k3 = thvec*b/sqrt(total(b^2))
      rt = r0 + b4*k1 + c4*k2 + d4*k3

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k4 = thvec*b/sqrt(total(b^2))
      rt = r0 + b5*k1 + c5*k2 + d5*k3 + e5*k4

      if outedge(rt) then begin
        out = 1
        break
      endif

      edgecheck, rt
      b = trilinear3d_grid(rt, bgrid)
      k5 = thvec*b/sqrt(total(b^2))
      
      rt = r0 + n1*k1 + n3*k3 + n4*k4 + n5*k5
      edgecheck, rt

      if outedge(rt) then begin
        out = 1
        break
      endif

      count++

      line.add, rt

      ; check line is still moving
      if (count ge 3) then begin
        dl = line[-1] - line[-2]
        mdl = sqrt(total(dl^2))
        if mdl lt hmin/2 then break

        dl = line[-1] - line[-3]
        mdl = sqrt(total(dl^2))
        if mdl lt hmin/2 then begin
          bounce = 1
          break
        endif
      endif

    end
    
    if bounce eq 1 then line = line[0:-2]

    if out eq 1 then begin
        rin = line[-1]
        rout = rt
      if rout[0] gt xmax or rout[0] lt xmin then begin
        if rout[0] gt xmax then xedge = xmax else xedge = xmin
        s = (xedge - rin[0])/(rout[0] - rin[0])
        rout = s*(rout - rin) + rin
      endif
      if rout[1] gt ymax or rout[1] lt ymin then begin
        if rout[1] gt ymax then yedge = ymax else yedge = ymin
        s = (yedge - rin[1])/(rout[1] - rin[1])
        rout = s*(rout - rin) + rin
      endif
      if rout[2] gt zmax or rout[2] lt zmin then begin
        if rout[2] gt zmax then zedge = zmax else zedge = zmin
        s = (zedge - rin[2])/(rout[2] - rin[2])
        rout = s*(rout - rin) + rin
      endif
      line.add, rout
    endif

    line.reverse

  endforeach

  if not keyword_set(gridcoord) then begin
    foreach pt, line, ipt do begin
      ix = floor(pt[0])
      iy = floor(pt[1])
      iz = floor(pt[2])
      if ix eq xmax then ix--
      if iy eq ymax then iy--
      if iz eq zmax then iz--

      line[ipt, 0] = x[ix] + (pt[0] - ix)*(x[ix+1] - x[ix])
      line[ipt, 1] = y[iy] + (pt[1] - iy)*(y[iy+1] - y[iy])
      line[ipt, 2] = z[iz] + (pt[2] - iz)*(z[iz+1] - z[iz])
    endforeach
  endif
  
  if keyword_set(oneway) then line.reverse

  return, transpose(line.toarray())

end
