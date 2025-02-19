common shared_var_read, outdir

function prefix, filename
  compile_opt idl2

    ; file1 = strmid(filename, strlen('data/'), strlen(filename))
    ; file1 = strmid(file1, 0, strlen(file1)-strlen('.dat'))
    ; dot = filename[::-1].find('.')
    ; slash = filename[::-1].find('/')

    dot = filename.lastindexof('.')
    slash = filename.lastindexof('/')
    len = filename.strlen()
    file1 = filename.substring(slash+1, dot-1)

    return, file1

end

function read_field, filename, grid=grid
  compile_opt idl2
  common shared_var_read

  openr, field, filename, /get_lun
  nx = 0L
  ny = 0L
  nz = 0L
  readu, field, nx, ny, nz
  bx = dblarr(nx, ny, nz)
  by = bx
  bz = bx
  x = dblarr(nx)
  y = dblarr(ny)
  z = dblarr(nz)
  readu, field, bx, by, bz, x, y, z
  close, field
  free_lun, field

  if keyword_set(grid) then begin
    bgrid = dblarr(nx, ny, nz, 3)
    bgrid[*, *, *, 0] = bx
    bgrid[*, *, *, 1] = by
    bgrid[*, *, *, 2] = bz
    return, {bgrid:bgrid, x:x, y:y, z:z}
  endif else begin
    return, {bx:bx, by:by, bz:bz, x:x, y:y, z:z}
  endelse

end

function read_nulls, filename, simple=simple
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'

  openr, null, outdir+'/'+prefix(filename)+'-nullpos.dat', /get_lun
  nnulls = 0L
  readu, null, nnulls
  if nnulls gt 0 then begin
    realpos = dblarr(3, nnulls)
    pos = realpos
    rp = dblarr(3)
    for i = 1, nnulls do begin
      readu, null, rp
      pos[*, i-1] = rp
    endfor
    for i = 1, nnulls do begin
      readu, null, rp
      realpos[*, i-1] = rp
    endfor
    close, null
    free_lun, null

    if not keyword_set(simple) then begin
      openr, null, outdir+'/'+prefix(filename)+'-nulldata.dat', /get_lun
      nnulls = 0L
      readu, null, nnulls

      signs = lonarr(nnulls)
      spine = dblarr(3, nnulls)
      fan = spine
      readu, null, signs, spine, fan

      close, null
      free_lun, null
    endif

    nulls = {nulldata, pos:dblarr(3), gridpos:dblarr(3), spine:dblarr(3), fan:dblarr(3), sign:0, number:0}
    nulls = replicate({nulldata}, nnulls)

    for i = 0, nnulls-1 do begin
      nulls[i].pos = realpos[*, i]
      nulls[i].gridpos = pos[*, i]
      nulls[i].number = i+1
      if not keyword_set(simple) then begin
        nulls[i].spine = spine[*, i]
        nulls[i].fan = fan[*, i]
        nulls[i].sign = signs[i]
      endif
    endfor
  endif

  return, nulls
  
end

function read_separators, filename, conlist=conlist, null_list=null_list, hcs=hcs
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'

  nulldata = read_nulls(filename, /simple)

  if not keyword_set(hcs) then begin
    openr, con, outdir+'/'+prefix(filename)+'-connectivity.dat', /get_lun
    openr, sep, outdir+'/'+prefix(filename)+'-separators.dat', /get_lun
    allnulls_list = nulldata.number
  endif else begin
    openr, con, outdir+'/'+prefix(filename)+'-hcs-connectivity.dat', /get_lun
    openr, sep, outdir+'/'+prefix(filename)+'-hcs-separators.dat', /get_lun
    allnulls_list = [1]
  endelse

  if not keyword_set(null_list) then null_list = allnulls_list

  conlist = list()
  seplist = list()

  endnum = 0L
  length = 0L
  nseps = 0L

  foreach inull, allnulls_list do begin
    coni = list()
    sepi = list()
    readu, con, nseps
    for i = 1, nseps do begin
      skip_lun, con, 4
      readu, con, endnum
      coni.add, endnum
      skip_lun, con, 8
      readu, sep, length
      separator = dblarr(3, length)
      readu, sep, separator
      sepi.add, separator
    endfor
    if where(inull eq null_list, /null) ne !null then begin
      conlist.add, coni
      seplist.add, sepi
    endif else begin
      conlist.add, list()
      seplist.add, list()
    endelse
    
  endforeach

  close, sep, con
  free_lun, sep, con

  return, seplist

end

function read_spines, filename, null_list=null_list
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'
  
  nulldata = read_nulls(filename, /simple)

  if not keyword_set(null_list) then null_list = nulldata.number

  spinelist = list()
  length = 0L

  openr, spi, outdir+'/'+prefix(filename)+'-spines.dat', /get_lun

  foreach inull, nulldata.number do begin
    spinelisti = list()
    for ispine = 1, 2 do begin
      readu, spi, length
      if where(inull eq null_list, /null) ne !null then begin
        spine = dblarr(3, length)
        readu, spi, spine
        spinelisti.add, spine
      endif else begin
        skip_lun, spi, 3LL*8LL*length
      endelse
    endfor
    spinelist.add, spinelisti
  endforeach

  close, spi
  free_lun, spi
  
  return, spinelist

end

function read_rings, filename, nskip=nskip, breaks=breaklist, assocs=assoclist, null_list=null_list, hcs=hcs
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'

  nulldata = read_nulls(filename, /simple)

  if not keyword_set(nskip) then nskip = 1

  if keyword_set(hcs) then begin
    assocs_filename = outdir+'/'+prefix(filename)+'-hcs-assocs.dat'
    info_filename = outdir+'/'+prefix(filename)+'-hcs-ringinfo.dat'
    ring_filename = outdir+'/'+prefix(filename)+'-hcs-rings.dat'
    break_filename = outdir+'/'+prefix(filename)+'-hcs-breaks.dat'
  endif else begin
    assocs_filename = outdir+'/'+prefix(filename)+'-assocs.dat'
    info_filename = outdir+'/'+prefix(filename)+'-ringinfo.dat'
    ring_filename = outdir+'/'+prefix(filename)+'-rings.dat'
    break_filename = outdir+'/'+prefix(filename)+'-breaks.dat'
  endelse

  if keyword_set(assocs) and not file_test(assocs_filename) then print, 'Not reading in associations: file does not exist'
  if keyword_set(assocs) and file_test(assocs_filename) then do_assocs = 1 else do_assocs = 0

  if nskip eq 1 and keyword_set(assocs) then print, 'Warning: Associations not corrected for when nskip ne 1. May be updated in future.'

  openr, rinfo, info_filename, /get_lun
  openr, ring, ring_filename, /get_lun
  openr, brks, break_filename, /get_lun
  if do_assocs then openr, ass, outdir+'/'+prefix(filename)+'-assocs.dat', /get_lun

  ringsmax = 0L
  writeskip = 0L
  bytesize = 0L
  stepsize = 0D
  n_hcs = 0L

  readu, rinfo, ringsmax, writeskip, bytesize, stepsize
  if keyword_set(hcs) then readu, rinfo, n_hcs
  if writeskip gt 1 then ringsmax = ceil(double(ringsmax)/writeskip)

  if keyword_set(hcs) then begin
    to_do = indgen(n_hcs/2) + 1
    ndir = 2
    name = 'hcs'
  endif else begin
    to_do = nulldata.number
    ndir = 1
    name = 'null'
  endelse

  if not keyword_set(null_list) and not keyword_set(hcs) then begin
    null_list = nulldata.number
  endif else if keyword_set(hcs) then begin
    null_list = to_do
  endif

  ringlist = list()
  breaklist = list()
  if do_assocs then assoclist = list()

  foreach inull, to_do do begin
    for dir = 1, ndir do begin
      breaklisti = list()
      ringlisti = list()
      if do_assocs then assoclisti = list()
      lengths = lonarr(ringsmax)
      readu, rinfo, lengths
      if where(inull eq null_list, /null) ne !null then begin
        foreach length, lengths[where(lengths ne 0)], iring do begin
          if iring mod nskip eq 0 then begin
            if do_assocs then assocs = lonarr(length)
            breaks = lonarr(length)
            if bytesize eq 4 then begin
              rings = fltarr(3, length)
            endif else if bytesize eq 8 then begin
              rings = dblarr(3, length)
            endif
            readu, ring, rings
            if do_assocs then readu, ass, assocs
            readu, brks, breaks
            breaklisti.add, breaks
            ringlisti.add, rings
            if do_assocs then assoclisti.add, assocs
          endif else begin
            skip_lun, ring, 3LL*bytesize*length
            skip_lun, brks, 4LL*length
            if do_assocs then skip_lun, ass, 4LL*length
          endelse
        endforeach
      endif else begin
        total_len = total(lengths, /integer)
        skip_lun, ring, 3LL*bytesize*total_len
        skip_lun, brks, 4LL*total_len
        if do_assocs then skip_lun, ass, 4LL*total_len
      endelse
      ringlist.add, ringlisti
      breaklist.add, breaklisti
      if do_assocs then assoclist.add, assoclisti
    endfor
  endforeach

  close, rinfo, ring, brks
  free_lun, rinfo, ring, brks
  if do_assocs then begin
    close, ass
    free_lun, ass
  endif
  
  return, ringlist

end

function get_cut_filename_plane, normal, d
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'

  if n_elements(d) eq 1 then begin
    d_pln = d
  endif else begin
    d_pln = np.sum(d*normal)
  endelse

  l = [normal, d_pln]
  fmt = '('
  fmt1 = 'f08.4'
  fmt2 = 'f09.4'
  foreach i, l, n do begin &$
    if i ge 0 then fadd = fmt1 else fadd = fmt2 &$
    fmt = fmt + fadd &$
    if n ne 3 then fmt = fmt + ', A, ' &$
  endforeach
  fmt = fmt + ')'

  return, string(l[0], '_', l[1], '_', l[2], '_', l[3], format=fmt)

end

function read_cut_sepsurf, filename, normal, d
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'
  
  filename_plane = get_cut_filename_plane(normal, d)
  null_nums = list()
  sepsurfs = list()
  
  nlines = 0L
  inull = 0L
  length = 0L

  openr, ringfile, outdir+'/'+prefix(filename)+'-rings-cut_'+filename_plane+'.dat', /get_lun
  readu, ringfile, nlines
  for i = 1, nlines do begin
    readu, ringfile, inull
    null_nums.add, inull
    readu, ringfile, length
    sepsurf = dblarr(3, length)
    readu, ringfile, sepsurf
    sepsurfs.add, sepsurf
  endfor
  close, ringfile
  free_lun, ringfile

  return, list(sepsurfs, null_nums)

end

function read_cut_separators, filename, normal, d, hcs=hcs
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'

  filename_plane = get_cut_filename_plane(normal, d)
  null_nums = list()
  sep_pts = list()
  npts = 0L
  endnum = 0L
  sep = dblarr(3)

  if keyword_set(hcs) then begin
    openr, sepfile, outdir+'/'+prefix(filename)+'-hcs-separators-cut_'+filename_plane+'.dat', /get_lun
    readu, sepfile, npts
    for i = 1, npts do begin
      readu, sepfile, endnum
      null_nums.add, endnum
      readu, sepfile, sep
      sep_pts.add, sep
    endfor
  endif else begin
    openr, sepfile, outdir+'/'+prefix(filename)+'-separators-cut_'+filename_plane+'.dat', /get_lun
    readu, sepfile, npts
    for i = 1, npts do begin
      readu, sepfile, endnum
      readu, sepfile, endnum
      null_nums.add, endnum
      readu, sepfile, sep
      sep_pts.add, sep
    endfor
  endelse
  close, sepfile
  free_lun, sepfile

  return, list(sep_pts, null_nums)

end

function read_cut_spines, filename, normal, d
  compile_opt idl2
  common shared_var_read
  
  if outdir eq !null then outdir = 'output'

  filename_plane = get_cut_filename_plane(normal, d)
  null_nums = list()
  spine_pts = list()

  npts = 0L
  inull = 0L
  spine = dblarr(3)

  openr, spinefile, outdir+'/'+prefix(filename)+'-spines-cut_'+filename_plane+'.dat', /get_lun
  readu, spinefile, npts
  for i = 1, npts do begin
    readu, spinefile, inull
    null_nums.add, inull
    readu, spinefile, spine
    spine_pts.add, spine
  endfor
  close, spinefile
  free_lun, spinefile

  return, list(spine_pts, null_nums)
  
end

function read_cut_hcs, filename, normal, d
  compile_opt idl2
  common shared_var_read

  if outdir eq !null then outdir = 'output'

  filename_plane = get_cut_filename_plane(normal, d)
  lines = list()

  nlines = 0L
  length = 0L
  temp = 0L
  
  openr, hcsfile, outdir+'/'+prefix(filename)+'-hcs-cut_'+filename_plane+'.dat', /get_lun
  readu, hcsfile, nlines
  for i = 1, nlines do begin
    readu, hcsfile, temp
    readu, hcsfile, length
    line = dblarr(3, length)
    readu, hcsfile, line
    lines.add, line
  endfor
  close, hcsfile
  free_lun, hcsfile

  return, lines

end

pro set_output_dir, loc
  common shared_var_read
  
  if loc.endswith('/') then loc = loc.remove(-1)
  outdir = loc
  print, 'Changed output data directory to be "./' + outdir + '"'

end
