function trilinear3d, xp, yp, zp, grid, xx, yy, zz
  compile_opt idl2

  ; inputs
  ; (xp, yp, zp) point at which field required in x, y, z units
  ; grid[nx, ny, nz, ncomp] - grid of field
  ; xx[nx], yy[ny], zz[nz] - x, y, z values

  nx = n_elements(xx)
  ny = n_elements(yy)
  nz = n_elements(zz)

  jacobi = dblarr(3, 3)

  ii = where(xx lt xp, nii)
  if (ii[0] eq -1) then ix = 0 else ix = ii[nii-1]
  if (ix eq nx-1) then ix = nx-2
  jj = where(yy lt yp, njj)
  if (jj[0] eq -1) then iy = 0 else iy = jj[njj-1]
  if (iy eq ny-1) then iy = ny-2
  kk = where(zz lt zp, nkk)
  if (kk[0] eq -1) then iz = 0 else iz = kk[nkk-1]
  if (iz eq nz-1) then iz = nz-2

  dx = (xp - xx[ix])/(xx[ix+1] - xx[ix])
  dy = (yp - yy[iy])/(yy[iy+1] - yy[iy])
  dz = (zp - zz[iz])/(zz[iz+1] - zz[iz])

  s = size(grid)

  if s(0) eq 3 then it = 0
  if s(0) eq 4 then it = 2

  bp = dblarr(it+1)

  for k=0, it do begin
    b000 = grid[ix, iy, iz, k]
    b100 = grid[ix+1, iy, iz, k]
    b010 = grid[ix, iy+1, iz, k]
    b110 = grid[ix+1, iy+1, iz, k]
    b001 = grid[ix, iy, iz+1, k]
    b101 = grid[ix+1, iy, iz+1, k]
    b011 = grid[ix, iy+1, iz+1, k]
    b111 = grid[ix+1, iy+1, iz+1, k]
    
    a = b000
    b = b100 - b000
    c = b010 - b000
    d = b110 - b100 - b010 + b000
    e = b001 - b000
    f = b101 - b100 - b001 + b000
    g = b011 - b010 - b001 + b000
    h = b111 - b110 - b101 - b011 + b100 + b010 + b001 - b000

    bp[k] = a + b*dx + c*dy + d*dx*dy + e*dz + f*dx*dz + g*dy*dz + h*dx*dy*dz
  endfor

  return, bp

end