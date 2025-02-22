function makefield, rs=rs,ps=ps

  nr = 200L
  np = 360L
  nt = 360L
  field = dblarr(nr,nt,np,3)
  
  ;grid limits
  rlim = [0,2d]
  plim = [0d,!dpi]
  tlim = [0d,!dpi] ;3d
  
  rdiff = rlim[1]-rlim[0]
  pdiff = plim[1]-plim[0]
  tdiff = tlim[1]-tlim[0] ;3d
  
  ;null positions
  r0 = [1d,1d]
  p0 = [0d,!dpi]
  
  sz = (size(field))[1:3]-1
  
  ;field parameters
  nullk = [1,-1]
  globalk = 1
  
  ;grids
  rs = dindgen(sz[0]+1)*rdiff/sz[0]+rlim[0]
  ps = dindgen(sz[2]+1)*pdiff/sz[1]+plim[0]
  ts = dindgen(sz[1]+1)*tdiff/sz[2]+tlim[0] ;3d
  
  ;make grid
  for ip = 0, sz[2] do begin
    for it = 0, sz[1] do begin ;3d
      for ir = 0, sz[0] do begin
        r = rs[ir]
        p = ps[ip]
        t = ts[it] ;3d
        dist = sqrt(r^2 + r0^2 - 2*r*r0*cos(p-p0))
        field[ir,it,ip,0] = total(nullk*(r-r0*cos(p-p0))/dist^3) + globalk*sin(p)
        field[ir,it,ip,2] = total(nullk*r0*sin(p-p0)/dist^3) + globalk*cos(p)
        field[ir,it,ip,1] = 0
      endfor
    endfor ;3d
  endfor
  
  openw, lun, "data/testfield.dat", /get_lun
    writeu, lun, nr, nt, np
    writeu, lun, field[*,*,*,0], field[*,*,*,1], field[*,*,*,2]
    writeu, lun, rs, ts, ps
  close, lun

  return, field

end