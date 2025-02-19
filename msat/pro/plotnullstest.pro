nullsfile = "data/nullpoints_2097.dat"

nullsme = getnullsgordon(/me)
;nullsgor = getnullsgordon(/gordon)
nullsand = getnulls(nullsfile)

openr,10,'data/newmag.dat'
  nx = 0l
  ny = 0l
  nz = 0l
  readu,10,nx,ny,nz
  bgrid = dblarr(nx,ny,nz,3)
  rg = dblarr(nx)
  tg = dblarr(ny)
  pg = dblarr(nz)
  readu,10,bgrid,rg,tg,pg
close,10

tg = tg[-1:0:-1]

bgrid = !null
nullpts = []

for i = 0, n_elements(nullsand)-1 do begin;
  r = nullsand[i].pos[0]
  t = !dpi/2 + nullsand[i].pos[2]
  p = nullsand[i].pos[1]
  
  ir = max(where(rg lt r))
  rgc = ir + (r - rg[ir])/(rg[ir+1]-rg[ir])
  it = max(where(tg lt t))
  tgc = it + (t - tg[it])/(tg[it+1]-tg[it])
  ip = max(where(pg lt p))
  pgc = ip + (p - pg[ip])/(pg[ip+1]-pg[ip])
  
  nullpts = [[nullpts],[rgc,tgc,pgc]]
endfor

nullpts = transpose(nullpts)

isign0 = where(nullsme.type eq 0)

xa = nullpts[*,0]
ya = nullpts[*,1]
za = nullpts[*,2]

;ra = nullsand.pos[0]
;ta = nullsand.pos[2]
;pa = nullsand.pos[1]

;xa = ra*cos(pa)*sin(ta)
;ya = ra*sin(pa)*sin(ta)
;za = ra*cos(ta)

foreach i, isign0 do begin
  
  null = nullsme[i]
  
  ;r0 = null.pos[0]
  ;t0 = null.pos[2]
  ;p0 = null.pos[1]
  
  ;x0 = r0*cos(p0)*sin(t0)
  ;y0 = r0*sin(p0)*sin(t0)
  ;z0 = r0*cos(t0)
  
  x0 = null.gridpos[0]-1
  y0 = null.gridpos[1]-1
  z0 = null.gridpos[2]-1
  
  dist = sqrt((xa-x0)^2 + (ya-y0)^2 + (za-z0)^2)
  mindist = min(dist,loc)
  
  if mindist gt 1d-3 then print, "warning", mindist
  
  print, mindist
  print, null.gridpos-1
  print, reform(nullpts[loc,*])
  ;print, null.pos
  ;print, nullsand[loc].pos
  ;print, nullsand[loc].type

  print,'.....................................'
  
endforeach

locs = []
count = 0

for i = 0, n_elements(nullsme)-1 do begin

  null = nullsme[i]
  
  ;r0 = null.pos[0]
  ;t0 = null.pos[2]
  ;p0 = null.pos[1]
  
  ;x0 = r0*cos(p0)*sin(t0)
  ;y0 = r0*sin(p0)*sin(t0)
  ;z0 = r0*cos(t0)
  
  x0 = null.gridpos[0]-1
  y0 = null.gridpos[1]-1
  z0 = null.gridpos[2]-1
  
  dist = sqrt((x0-xa)^2 + (y0-ya)^2 + (z0-za)^2)
  
  mindist = min(dist,loc)
  
  locs = [locs,loc]
  
  if mindist gt 1d-3 then print, i, mindist, loc
  if nullsme[i].type ne nullsand[loc].type and nullsme[i].type ne 0 then begin
    print, i+1, nullsme[i].type, loc, nullsand[loc].type
    count=count+1
  endif
  if not (floor(x0) eq floor(xa[loc]) and floor(y0) eq floor(ya[loc]) and floor(z0) eq floor(za[loc])) then print, "different cells"
  
endfor

print, "nulls different", count

for i = 0, n_elements(locs)-2 do begin
  for j = i+1, n_elements(locs)-1 do begin
    if locs[i] eq locs[j] then print, "same"
  endfor
endfor

end
