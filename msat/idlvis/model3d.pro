common shared_var_model3d, bgrid, xx, yy, zz, oModel, nulldata, ds, filename, nulllist, csystem_model3d, nskip_global, periodic_check, periodic_dist, periodic_global

function sphr2cart, pts
  compile_opt idl2

  carts = pts
  carts[0, *] = pts[0, *]*sin(pts[1, *])*cos(pts[2, *])
  carts[1, *] = pts[0, *]*sin(pts[1, *])*sin(pts[2, *])
  carts[2, *] = pts[0, *]*cos(pts[1, *])

  return, carts

end

function cyl2cart, pts
  compile_opt idl2

  carts = pts
  carts[0, *] = pts[0, *]*cos(pts[1, *])
  carts[1, *] = pts[0, *]*sin(pts[1, *])
  carts[2, *] = pts[2, *]

  return, carts

end

pro model_add_sepsurf, nskip=nskip, draw=draw, nlines=nlines, nring=nring
  common shared_var_model3d
  compile_opt idl2

  if not keyword_set(draw) then draw = 'rings'
  if not keyword_set(nskip) then nskip = 20

  if draw eq 'rings' then begin
    print, 'Adding separatrix surface rings'
    
    rings = read_rings(filename, breaks=breaks, nskip=nskip, null_list=nulllist)

    foreach inull, nulllist-1 do begin

      if nulldata[inull].sign gt 0 then colour = [255, 128, 128] else begin
        if nulldata[inull].sign lt 0 then colour = [128, 128, 255] else colour = [128, 255, 128]
      endelse

      for iring = 0, n_elements(rings[inull])-1 do begin
        ring = rings[inull, iring]
        if periodic_check then begin
          dists = total((ring[*, 1:-1] - ring[*, 0:-2])^2, 1)
          brks = [-1, where(breaks[inull, iring] eq 1, /null), where(dists gt periodic_dist), n_elements(breaks[inull, iring])-1]
        endif else begin
          brks = [-1, where(breaks[inull, iring] eq 1, /null), n_elements(breaks[inull, iring])-1]
        endelse
        brks = brks[uniq(brks)]
        if csystem_model3d eq 'spherical' then ring = sphr2cart(ring)
        for ib = 0, n_elements(brks)-2 do begin
          oModel.add, obj_new("IDLgrPolyline", ring[0, brks[ib]+1:brks[ib+1]], ring[1, brks[ib]+1:brks[ib+1]], ring[2, brks[ib]+1:brks[ib+1]], color=colour)
        endfor
      endfor
      
    endforeach
  endif else if draw eq 'fieldlines' then begin
    
    if not keyword_set(nlines) then nlines = 50

    rings = read_rings(filename, nskip=nskip, null_list=nulllist)
    
    foreach inull, nulllist-1 do begin

      if nulldata[inull].sign gt 0 then colour = [255, 128, 128] else begin
        if nulldata[inull].sign lt 0 then colour = [128, 128, 255] else colour = [128, 255, 128]
      endelse

      if not keyword_set(nring) then begin
        ring = rings[inull, n_elements(rings[inull])/5]
      endif else begin
        if typename(nring) eq typename(1) or typename(nring) eq typename(1l) then begin
          ring = rings[inull, nring]
        endif else if typename(nring) eq typename(0.1) or typename(nring) eq typename(0.1d) then begin
          ring = rings[inull, floor(nring*n_elements(rings[inull]))]
        endif
      endelse

      nskip = n_elements(ring[0, *])/nlines

      for i = 0, nlines-1 do begin
        
        startpt = ring[*, i*nskip]
        h = 1d-2
        hmin = 1d-3
        hmax = 0.5
        epsilon = 1d-5

        line = fieldline3d(startpt, bgrid, xx, yy, zz, h, hmin, hmax, epsilon, coordsystem=csystem_model3d)

        dists = sqrt((line[0, *] - nulldata[inull].pos[0])^2 + $
                (line[1, *] - nulldata[inull].pos[1])^2 + $
                (line[2, *] - nulldata[inull].pos[2])^2)
        a = min(dists, imin)
        if nulldata[inull].sign eq -1 then line = line[*, 0:imin] else line = line[*, imin:-1]

        if csystem_model3d eq 'spherical' then line = sphr2cart(line)

        if periodic_check then begin
            dists = total((ring[*, 1:-1] - ring[*, 0:-2])^2, 1)
            brks = [-1, where(dists gt 0.9*periodic_dist), n_elements(ring)/3-1]
            for ib = 0, n_elements(brks)-2 do begin
              oModel.add, obj_new("IDLgrPolyline", line[0, brks[ib]+1:brks[ib+1]], line[1, brks[ib]+1:brks[ib+1]], line[2, brks[ib]+1:brks[ib+1]], color=colour)
            endfor
          endif else begin
            oModel.add, obj_new("IDLgrPolyline", line[0, *], line[1, *], line[2, *], color=colour)
          endelse

      endfor

    endforeach
  endif else print, "Set draw to be either 'rings' or 'fieldlines'"
end

pro model_add_hcs, nskip=nskip, nlines=nlines, draw=draw
  common shared_var_model3d
  compile_opt idl2

  if not keyword_set(draw) then draw = 'rings'
  if not keyword_set(nlines) then nlines = 100

  colour = [0, 255, 0]

  if draw eq 'rings' then begin

    rings = read_rings(filename, breaks=breaks, nskip=nskip, /hcs)

    for inull = 0, n_elements(rings)-1 do begin

      for iring = 0, n_elements(rings[inull])-1 do begin
        brks = [-1, where(breaks[inull, iring] eq 1, /null), n_elements(breaks[inull, iring])-1]
        brks = brks[uniq(brks)]
        ring = rings[inull, iring, *, *]
        ring = sphr2cart(ring)
        for ib = 0, n_elements(brks)-2 do begin
          oModel.add, obj_new("IDLgrPolyline", ring[0, brks[ib]+1:brks[ib+1]], ring[1, brks[ib]+1:brks[ib+1]], ring[2, brks[ib]+1:brks[ib+1]], color=colour)
        endfor
      endfor
      
    endfor

  endif else if draw eq 'fieldlines' then begin

    rings = read_rings(filename, nskip=nskip, /hcs)

    for inull = 0, n_elements(rings)-1, 2 do begin

      iring = 1
      nskip = n_elements(rings[inull, iring, 0, *])/nlines

      for idir = 0, 1 do begin

        for i = 0, nlines-1 do begin

          startpt = rings[inull+idir, iring, *, i*nskip]

          h = 1d-2
          hmin = h*0.1
          hmax = h*10
          epsilon = h*0.01

          line = fieldline3d(startpt, bgrid, xx, yy, zz, h, hmin, hmax, epsilon, coordsystem=csystem_model3d)

          a = max(line[0, *], imax)
          if idir eq 1 then begin
            line = line[*, 0:imax]
          endif else begin
            line = line[*, imax:-1]
          endelse
          
          line = sphr2cart(line)

          oModel.add, obj_new("IDLgrPolyline", line[0, *], line[1, *], line[2, *], color=colour)

        endfor

      endfor

    endfor
  
  endif else print, "Set draw to be either 'rings' or 'fieldlines'"

  for inull = 0, n_elements(rings)-1, 2 do begin
    ring = rings[inull, 0]
    ring = sphr2cart(ring)
    oModel.add, obj_new("IDLgrPolyline", ring[0, *], ring[1, *], ring[2, *], color=colour, thick=4)
  endfor

end

pro model_add_spines
  common shared_var_model3d
  compile_opt idl2

  print, 'Adding Spines'

  spines = read_spines(filename, null_list=nulllist)

  foreach inull, nulllist-1 do begin
    if nulldata[inull].sign gt 0 then col = [250, 0, 0] else col = [0, 0, 250]
    for dir = 0, 1 do begin
      
      spine = spines[inull, dir, *, *]
      if csystem_model3d eq 'spherical' then spine = sphr2cart(spine)

      if periodic_check then begin
        dists = total((spine[*, 1:-1] - spine[*, 0:-2])^2, 1)
        brks = [-1, where(dists gt 0.9*periodic_dist), n_elements(spine)/3-1]
        brks = brks[uniq(brks)]
        for ib = 0, n_elements(brks)-2 do begin
          oModel.add, obj_new("IDLgrPolyline", spine[0, brks[ib]+1:brks[ib+1]], spine[1, brks[ib]+1:brks[ib+1]], spine[2, brks[ib]+1:brks[ib+1]], color=col, thick=4)
        endfor
      endif else begin
        oModel.add, obj_new("IDLgrPolyline", spine[0, *], spine[1, *], spine[2, *], color=col, thick=4)
      endelse

    endfor
  endforeach
end

pro model_add_separators, hcs=hcs
  common shared_var_model3d
  compile_opt idl2

  print, 'Adding separators'

  if keyword_set(hcs) then colour = [255, 165, 0] else colour = [255, 255, 0]

  seps = read_separators(filename, null_list=nulllist, hcs=hcs)

  if keyword_set(hcs) then to_do = [0] else to_do = nulllist-1

  foreach inull, to_do do begin
    
    for isep = 0, n_elements(seps[inull])-1 do begin
      sep = seps[inull, isep, *, *]
      if csystem_model3d eq 'spherical' then sep = sphr2cart(sep)

      if periodic_check then begin
        dists = total((sep[*, 1:-1] - sep[*, 0:-2])^2, 1)
        brks = [-1, where(dists gt 0.9*periodic_dist), n_elements(line)/2-1]
        brks = brks[uniq(brks)]
        for ib = 0, n_elements(brks)-2 do begin
          oModel.add, obj_new("IDLgrPolyline", sep[0, brks[ib]+1:brks[ib+1]], sep[1, brks[ib]+1:brks[ib+1]], sep[2, brks[ib]+1:brks[ib+1]], color=colour, thick=4)
        endfor
      endif else begin
        oModel.add, obj_new("IDLgrPolyline", sep[0, *], sep[1, *], sep[2, *], color=colour, thick=4)
      endelse

    endfor

  endforeach
end

pro model_add_nulls, size=size
  common shared_var_model3d
  compile_opt idl2

  boxsize = min([xx[-1] - xx[0], yy[-1] - yy[0], zz[-1] - zz[0]])/40
  if not keyword_set(size) then size = 1
  r = max([boxsize, ds])*size
  radius = ds
  
  foreach inull, nulllist-1 do begin
    mesh_obj, 4, vert, poly, replicate(radius,21,21)
    pos = nulldata[inull].pos
    if csystem_model3d eq 'spherical' then pos = [pos[0]*sin(pos[1])*cos(pos[2]), pos[0]*sin(pos[1])*sin(pos[2]), pos[0]*cos(pos[1])]
    for j = 0, 2 do vert[j, *] = vert[j, *] + pos[j]
    if nulldata[inull].sign gt 0 then colour = [255, 0, 0] else begin
      if nulldata[inull].sign lt 0 then colour = [0, 0, 255] else colour = [0, 255, 0]
    endelse
    oModel.add, obj_new("IDLgrPolygon", data=vert, polygons=poly, color=colour, /shading)
  endforeach
  
end

pro model_add_fieldlines, startpts, colour=colour
  common shared_var_model3d
  compile_opt idl2

  if not keyword_set(colour) then colour = [0, 0, 0]

  for i = 0, n_elements(startpts)/3 - 1  do begin
        
    startpt = startpts[*, i]
    h = 1d-2
    hmin = 1d-3
    hmax = 0.5
    epsilon = 1d-5

    line = fieldline3d(startpt, bgrid, xx, yy, zz, h, hmin, hmax, epsilon, coordsystem=csystem_model3d)

    if csystem_model3d eq 'spherical' then line = sphr2cart(line)

    if periodic_check then begin
      dists = total((line[*, 1:-1] - line[*, 0:-2])^2, 1)
      brks = [-1, where(dists gt 0.9*periodic_dist), n_elements(line)/3-1]
      brks = brks[uniq(brks)]
      for ib = 0, n_elements(brks)-2 do begin
        oModel.add, obj_new("IDLgrPolyline", line[0, brks[ib]+1:brks[ib+1]], line[1, brks[ib]+1:brks[ib+1]], line[2, brks[ib]+1:brks[ib+1]], color=colour)
      endfor
    endif else begin
      oModel.add, obj_new("IDLgrPolyline", line[0, *], line[1, *], line[2, *], color=colour)
    endelse

  endfor

end

pro model_add_box
  common shared_var_model3d
  compile_opt idl2
  
  print, "Adding box"
  box = dblarr(3,2)
  box[*,0] = [min(xx), min(yy), min(zz)] 
  box[*,1] = [max(xx), max(yy), max(zz)]
  line = double(transpose([[0, 0, 0], [1, 0, 0], [1, 0, 1], $
      [1, 0, 0], [1, 1, 0], [1, 1, 1], $
      [1, 1, 0], [0, 1, 0], [0, 1, 1], $
      [0, 1, 0], [0, 0, 0], [0, 0, 1], $
      [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]]))
  
  line[*, 0] = line[*, 0]*(box[0, 1] - box[0, 0]) + box[0, 0]
  line[*, 1] = line[*, 1]*(box[1, 1] - box[1, 0]) + box[1, 0]
  line[*, 2] = line[*, 2]*(box[2, 1] - box[2, 0]) + box[2, 0]

  dist = min([n_elements(xx), n_elements(yy), n_elements(zz)])*ds/5
  oModel.add, obj_new('idlgrtext', 'x', locations=[box[0, 0]+dist, box[1, 0], box[2, 0]], /onglass)
  oModel.add, obj_new('idlgrtext', 'y', locations=[box[0, 0], box[1, 0]+dist, box[2, 0]], /onglass)
  oModel.add, obj_new('idlgrtext', 'z', locations=[box[0, 0], box[1, 0], box[2, 0]+dist], /onglass)
  
  oModel.add, obj_new('IDLgrPolyline', line[*, 0], line[*, 1], line[*, 2], color=[0, 0, 0])
end

pro model_add_sun, absbrmax=absbrmax
  common shared_var_model3d
  compile_opt idl2

  if not keyword_set(absbrmax) then absbrmax = 10

  br_sun = reform(bgrid[0, *, *, 0])
  brsun = transpose(br_sun)
  nsz = size(brsun)
  np = nsz[1]
  nt = nsz[2]
  br_tsun = brsun
  for k = 0, nt-1 do br_tsun[*, k] = brsun[*, nt-1-k]
  brmax = max(abs(br_tsun)) 
  vc = float(br_tsun)
  ii = where(br_tsun gt brmax*0.25)
  vc[ii] = brmax*0.25
  ii = where(br_tsun lt -brmax*0.25)
  vc[ii] = -brmax*0.25
  vc = fix(((vc+brmax*0.25)/(brmax*0.5))*255)

  vert_colour = fltarr(3, np*nt)
  for i = 0, 2 do begin
    for j = 0, nt-1 do begin
      vert_colour[i, j*np:(j+1)*np-1] = vc[*, j]
    endfor
  endfor
  
  mesh_obj, 4, vert, poly, replicate(1.00, np, nt)
  oModel.add, obj_new("IDLgrPolygon", data=vert, vert_colors=vert_colour, polygons=poly, shading=0)

  mesh_obj, 3, vert, poly, replicate(0.02, 21, 2), p3=-2.4, p4=2.4
  oModel.add, obj_new("IDLgrPolygon", data=vert, polygons=poly, color=[102, 0, 102])

  mesh_obj, 3, vert, poly, (findgen(3)/2.*0.08) ## replicate(1, 21), p3=2.5, p4=2.4
  oModel.add, obj_new("IDLgrPolygon", data=vert, polygons=poly, color=[102, 0, 102])

end

pro save, dimensions=dimensions

  xobjview_write_image, 'figures/' + rd.prefix(filename) + '-model3d.png', 'png', dimensions=dimensions

end

pro set_null_list, lst
  common shared_var_model3d
  compile_opt idl2

  if lst eq !null then begin
    nulllist = nulldata.number
  endif else begin
    if max(lst) gt max(nulldata.number) or min(lst) < 1 then print, 'Invalid list'
    nulllist = lst
  endelse
end

function make_model, fname, nulls=nulls, separators=separators, sepsurf=sepsurf, hcs=hcs, hcs_separators=hcs_separators, spines=spines, box=box, fanlines=fanlines, nskip=nskip, null_list=null_list, coordsystem=coordsystem, periodicity=periodicity
  common shared_var_model3d
  compile_opt idl2

  if not keyword_set(coordsystem) then coordsystem = 'cartesian'
  csystem_model3d = coordsystem

  filename = fname

  field = read_field(fname, /grid)
  bgrid = field.bgrid
  xx = field.x
  yy = field.y
  zz = field.z
  field = !null

  ds = min([min(xx[1:-1] - xx[0:-2]), min(yy[1:-1] - yy[0:-2]), min(zz[1:-1] - zz[0:-2])])

  periodic_check = 0
  if keyword_set(periodicity) then begin
    periodic_dist = 1d100
    periodic_check = 1
    if periodicity.contains('x') then periodic_dist = min([periodic_dist, (xx[-1] - xx[0])^2])
    if periodicity.contains('y') then periodic_dist = min([periodic_dist, (yy[-1] - yy[0])^2])
    if periodicity.contains('z') then periodic_dist = min([periodic_dist, (zz[-1] - zz[0])^2])
  endif
  if keyword_set(periodicity) then periodic_global = periodicity

  nulldata = read_nulls(filename)
  set_null_list, null_list

  oModel = obj_new("IDLgrModel")
  if keyword_set(box)            then model_add_box
  if keyword_set(nulls)          then model_add_nulls
  if keyword_set(sepsurf_flines) then model_add_sepsurf, nskip=nskip, draw='fieldlines'
  if keyword_set(spines)         then model_add_spines
  if keyword_set(sepsurf_rings)  then model_add_sepsurf, nskip=nskip, draw='rings'
  if keyword_set(hcs_rings)      then model_add_hcs, nskip=nskip, draw='rings'
  if keyword_set(hcs_flines)     then model_add_hcs, nskip=nskip, draw='fieldlines'
  if keyword_set(hcs_separators) then model_add_separators, /hcs
  if keyword_set(separators)     then model_add_separators
  
  return, oModel
end
