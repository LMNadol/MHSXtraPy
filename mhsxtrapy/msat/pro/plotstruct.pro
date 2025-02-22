function convert, gridcoord, x, y, z, check=check

  r = gridcoord[0]
  t = gridcoord[1]
  p = gridcoord[2]
  
  r = x[floor(r)] + (r-floor(r))(x[ceil(r)]-x[floor(r)])
  t = y[floor(t)] + (t-floor(t))(y[ceil(t)]-y[floor(t)])
  p = z[floor(p)] + (p-floor(p))(z[ceil(p)]-z[floor(p)])
  if keyword_set(check) then print, r,t,p
  return, [r*cos(p)*sin(t), r*sin(p)*sin(t), r*cos(t)]

end

pro plotstruct, n, converge=converge, fan=fan, ball=ball, rsphere=rsphere, file=file

  !except = 0
  
  if not keyword_set(file) then file = 'data/magfield.dat'; else file = 'data/' + file
  openr,10,file
  nx = 0l
  ny = 0l
  nz = 0l
  readu,10,nx,ny,nz
  bgrid = dblarr(nx,ny,nz,3)
  x = dblarr(nx)
  y = dblarr(ny)
  z = dblarr(nz)
  readu,10,bgrid,x,y,z
  close,10
  
  sizebg = size(bgrid)
  xgc = indgen(sizebg[1])
  ygc = indgen(sizebg[2])
  zgc = indgen(sizebg[3])

  nulls = read_nulls(file, /simple)
  npos = nulls[n-1].gridpos-1 ; -1 translation from fortran to idl grid spacing

  ;xyznpos = convert(npos,x,y,z)
  xyznpos = npos
  plt = plot3d([xyznpos[0]],[xyznpos[1]],[xyznpos[2]],'red',lines=' ',sym='x', sym_thick=3, sym_size=1.5, dim=[1200,900])
  plt.scale,1.5,1.5,1.5
  
  if not keyword_set(rsphere) then rsphere = 1d-3 ; rsphere from sf_converge/params.f90

  if keyword_set(converge) then begin
    ny=0l
    spine = dblarr(3)
    maxvec = spine
    minvec = spine
    fname = 'output/'+['maxvec.dat', 'minvec.dat', 'spine.dat']

    foreach name, fname do begin
      openr,10,name
      readu,10,ny
      r = dblarr(3,ny)
      readu,10,r
      help,r
      if name eq fname[0] then begin
        r = transpose(r)
        r = r * rsphere
        for i = 0, 2 do r[*,i] = r[*,i] + npos[i]
        ;for i = 0, n_elements(r)/3-1 do r[i,*] = convert(r[i,*],x,y,z)
        p = plot3d(r[*,0],r[*,1],r[*,2],lines=' ',sym='x', aspect_ratio=1,/overplot)
        
        if not eof(10) then begin
          readu,10,maxvec
          max1 = maxvec
          maxvec = maxvec*rsphere
          maxvec = maxvec + npos
          ;maxvec = convert(maxvec,x,y,z)
          p = plot3d([maxvec[0]],[maxvec[1]],[maxvec[2]],'cyan',lines=' ',sym='x',/overplot, sym_thick=6, sym_size=3,/aspect_ratio)
        endif
        if 1 eq 0 then begin
          if not eof(10) then begin
            readu, 10, n & crossfan = dblarr(3,n) & readu, 10, crossfan & help, crossfan
            print, crossfan
            crossfan = transpose(crossfan)
            crossfan = crossfan * rsphere
            for i = 0, 2 do crossfan[*,i] = crossfan[*,i] + npos[i]
            ;for i = 0, n_elements(crossfan)/3-1 do crossfan[i,*] = convert(crossfan[i,*],x,y,z)
            p = plot3d(reform(crossfan[0,*]),reform(crossfan[1,*]),reform(crossfan[2,*]),'orange',lines=' ',sym='x',/overplot, sym_thick=3, sym_size=2)
            cross = dblarr(3,n_elements(r)-1)
            ;for i = 1, n_elements(r)/3-1 do begin
            ;  cross[*,i-1] = crossp(r[*,0],r[*,i])
            ;  cross[*,i-1] = cross[*,i-1]/sqrt(cross[0,i-1]^2+cross[1,i-1]^2+cross[2,i-1]^2)
            ;endfor
            ;print, cross
            ;p = plot3d(reform(cross[0,*]),reform(cross[1,*]),reform(cross[2,*]),lines=' ',sym='x',/overplot, sym_thick=4, sym_size=3)
          endif
        endif
      endif
      if name eq fname[1] then begin
        r = transpose(r)
        r = r * rsphere
        for i = 0, 2 do r[*,i] = r[*,i] + npos[i]
        ;for i = 0, n_elements(r)/3-1 do r[i,*] = convert(r[i,*],x,y,z)
        p = plot3d(r[*,0],r[*,1],r[*,2],'g',lines=' ',sym='x', aspect_ratio=1,/overplot)
        
        if not eof(10) then begin
          readu,10,minvec
          min1 = minvec
          minvec = minvec*rsphere
          minvec = minvec + npos
          ;minvec = convert(minvec,x,y,z)
          p = plot3d([minvec[0]],[minvec[1]],[minvec[2]],'g',lines=' ',sym='x',/overplot, sym_thick=6, sym_size=3,/aspect_ratio)
        endif
      endif
      if name eq fname[2] then begin
        r = transpose(r)
        r = r * rsphere
        for i = 0,2 do r[*,i] = r[*,i] + npos[i]
        ;for i = 0, n_elements(r)/3-1 do r[i,*] = convert(r[i,*],x,y,z)
        p = plot3d(r[*,0],r[*,1],r[*,2],'r',lines=' ',sym='x',/overplot,/aspect_ratio)
        
        if not eof(10) then begin
          readu, 10, spine
          spine = spine*rsphere
          spine = spine + npos
          ;spine = convert(spine,x,y,z)
          p = plot3d([spine[0]],[spine[1]],[spine[2]],'blue',lines=' ',sym='x',/overplot, sym_thick=6, sym_size=3,/aspect_ratio)
        endif
      endif
      close,10
    endforeach
  endif
  
  boxedge = dblarr(2,3)
  for i = 0, 2 do boxedge[*,i] = [npos[i] - rsphere, npos[i] + rsphere]
  
  startpts = []
  if keyword_set(fan) then begin
    rad = 0.6*rsphere
    np = 20
    phi = 2*!dpi*((dindgen(np)+0.5)/np)
    for i = 0, np-1 do begin
      startpts = [[startpts],[max1*cos(phi[i])+min1*sin(phi[i])]]
    endfor
    startpts = startpts*rad
  endif
  
  if keyword_set(ball) then begin
    rad = 5*rsphere/2d1
    nt = 6
    np = 6
    theta = !dpi*((dindgen(nt)+0.5)/nt)
    phi = 2*!dpi*((dindgen(np)+0.5)/np)
    for i = 0, nt-1 do begin
      for j = 0, np-1 do begin
        xp = rad*cos(phi[j])*sin(theta[i])
        yp = rad*sin(phi[j])*sin(theta[i])
        zp = rad*cos(theta[i])
        startpts = [[startpts],[xp,yp,zp]]
      endfor
    endfor
  endif

  if keyword_set(ball) or keyword_set(fan) then begin
    for i = 0, 2 do startpts[i,*] = startpts[i,*] + npos[i]
    
    h0 = rsphere*1d-3
    for j = 0, n_elements(startpts)/3-1 do begin
      h = h0
      line = fieldline3d(startpts[*,j], bgrid, xgc,ygc,zgc, h, 0.1d*h, 10d*h, 0.01d*h, boxedge=boxedge, /oneway, /gridcoord)
      line = transpose(line)
      ;for i = 0, n_elements(line)/3-1 do line[i,*] = convert(line[i,*],x,y,z)
      plt = plot3d(line[*,0],line[*,1],line[*,2],'red',/overplot,/aspect_ratio,/perspective)
      
      h = h0
      line = fieldline3d(startpts[*,j], bgrid, xgc,ygc,zgc, -h, 0.1d*h, 10d*h, 0.01d*h, boxedge=boxedge, /oneway, /gridcoord)
      line = transpose(line)
      ;for i = 0, n_elements(line)/3-1 do line[i,*] = convert(line[i,*],x,y,z)
      plt = plot3d(line[*,0],line[*,1],line[*,2],'orange',/overplot,/aspect_ratio)
    endfor
  endif
  
  if 0 eq 1 then begin
    dist = 1d-1
    angle = acos(1d - 0.5*dist^2)
    totalangle = 0d
    pts = []
    while totalangle lt 2*!dpi do begin
      pts = [[pts],[max1*cos(totalangle)+min1*sin(totalangle)]]
      totalangle = totalangle + angle
    endwhile
    pts = transpose(pts)
    pts = pts * rsphere
    for i = 0, 2 do pts[*,i] = pts[*,i] + npos[i]
    ;for i = 0, n_elements(pts)/3-1 do pts[i,*] = convert(pts[i,*],x,y,z)
    plt = plot3d(pts[*,0],pts[*,1],pts[*,2],linest=' ',sym='x',/overplot,sym_thick=2, sym_size=2,/aspect_ratio)
  endif
  
  plt.xtitle = "x"
  plt.ytitle = "y"
  plt.ztitle = "z"
  
  print, "----------------------------------------"
  print, "Key:
  print, "----"
  print, "Spine - Dark blue cross"
  print, "Maxvec - Cyan cross"
  print, "Minvec - Green cross"
  print, "Null - Red cross"
  print, "Converged fan - Small black crosses"
  print, "Converged spine - Small red crosses"
  print, "Lines traced forwards - Red lines"
  print, "Lines traced backwards - Orange Lines"
  print, "----------------------------------------"
    
end
