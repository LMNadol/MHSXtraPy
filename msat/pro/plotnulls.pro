openr, 1, 'output/null.dat', /f77
  nnulls = 0l
  readu, 1, nnulls
  realpos = dblarr(3,nnulls) & pos = realpos
  rp = dblarr(3)
  for i = 1, nnulls do begin
    readu,1,rp
    pos[*,i-1] = rp
  endfor
  for i = 1, nnulls do begin
    readu,1,rp
    realpos[*,i-1] = rp
  endfor
close, 1

openr, 1, 'output/nullsben.dat', /f77
  readu, 1, nnulls
  signs = lonarr(nnulls)
  r = dblarr(3,nnulls)
  spine = dblarr(3,nnulls)
  fan = spine
  readu,1,signs
  readu,1,r,spine,fan
  signs = -1*signs

close,1

nulls = {nulldata,pos:dblarr(3),gridpos:dblarr(3),spine:dblarr(3),fan:dblarr(3),type:long(0)}
nulls = replicate({nulldata},nnulls)

for i = 0, nnulls-1 do begin
  nulls[i].pos = realpos[*,i]
  nulls[i].gridpos = pos[*,i] 
  nulls[i].spine = spine[*,i]
  nulls[i].fan = fan[*,i]
  nulls[i].type = signs[i]
endfor

print, "pos", n_elements(where(nulls.type eq 1)), "neg", n_elements(where(nulls.type eq -1))

radii = realpos[0,*]-1
sorted = sort(radii)
nullsold = nulls
nulls = nulls[sorted]

p = plot([0],[0],/nodata,/buffer,dimension=[1600,800],title='Ordered nulls with separator connections', xtit='Null number', ytit='Height', ylog=1, font_size=20, xstyle=3, ystyle=3)
;plot_io, [0,2300],[0,1.5]
;stop

;p = plot(radii[sorted], xstyle=1, yr=[0.9,1.8], thick=3)

lmax=351
nrad=206
rg=exp((!pi/(2.*(lmax+1.0)))*dindgen(nrad*1.))

foreach isort, sorted, inull do begin
  if inull gt -1 then begin
    istr = string(isort,format='(I4.4)')
    sepname = 'output/separator'+istr+'.dat'
    ringfiles = file_search('output/everything'+istr+'-????.dat')
    finf = file_info(sepname)
    nseps = (finf.size/4 - 1)/5
    print, inull, "Null ", isort, ", r =", nulls[isort].pos[0], ", Number of seps", nseps
    if nseps gt 0 then begin
      if file_test('output/sepnull'+istr+'.sav') then restore, 'output/sepnull'+istr+'.sav' else begin
        print, "calculating"
        sepdat = lonarr(5)
        openr, 1, sepname
        ;print, "Opened " + sepname
        sepmaxrad = dblarr(nseps)
        null2 = intarr(nseps)
        for isep = 1, nseps do begin
          readu, 1, sepdat
          ;print, sepdat
          linerad = dblarr(sepdat[3])
          for iring = sepdat[3], 1, -1 do begin
            dataname = 'output/everything'+istr+'-'+string(iring, format='(I4.4)')+'.dat'
            openr, 2, dataname
            ;print, "Opened " + dataname
            npoints = 0L
            readu, 2, npoints
            ;print, npoints
            r = dblarr(3,npoints)
            readu, 2, r
            assoc = lonarr(npoints)
            readu, 2, assoc
            close, 2
            assoc = assoc-1
            if iring eq sepdat[3] then begin
              linerad[iring-sepdat[3]] = r[0,sepdat[4]-1]
              nextindex = assoc[sepdat[4]-1]
            endif else begin
              linerad[iring-sepdat[3]] = r[0,nextindex]
              ;print, r[0,nextindex]
              nextindex = assoc[nextindex]
            endelse
          endfor
          sepmaxrad[isep-1] = max(linerad)
          ;print, "max found", sepmaxrad[isep-1]
          ;wait, 0.3
          null2[isep-1] = sepdat[2]-1
        endfor
        ;print, sepmaxrad
        ;print, null2
        nulluniq = uniq(null2)
        sepmaxrad2 = dblarr(n_elements(nulluniq))
        foreach jnull, nulluniq, j do sepmaxrad2[j] = max(sepmaxrad(where(null2 eq jnull)))
        save, nulluniq, sepmaxrad2, filename='output/sepnull'+istr+'.sav'
      endelse
      count=0
      foreach rad, sepmaxrad2, irad do begin
        jrad = floor(rad)-1
        rad1 = rg(jrad) + (rad-floor(rad))*(rg[jrad+1]-rg[jrad]) - 1
        ;print, rad
        newnull1 = inull
        newnull2 = where(sorted eq nulluniq[irad])
        if rad1 gt nulls[newnull1].pos[0]-1 and rad1 gt nulls[newnull2].pos[0]-1 then begin
          p = plot([newnull1,newnull1],[nulls[newnull1].pos[0]-1,rad1],'--',/overplot)
          ;oplot, [newnull1,newnull1],[nulls[newnull1].pos[0]-1,rad1], linestyle=2
          p = plot([newnull2,newnull2],[nulls[newnull2].pos[0]-1,rad1],'--',/overplot)
          ;oplot, [newnull2,newnull2],[nulls[newnull2].pos[0]-1,rad1], linestyle=2
          p = plot([newnull1,newnull2],[rad1,rad1],/overplot)
          ;oplot, [newnull1,newnull2],[rad1,rad1]
          count++
        endif; else print, "warning"
      endforeach
      print, "plotted", count, "seps with max height", max(sepmaxrad2)
      close, 1
    endif
  endif
endforeach

ip = where(nulls.type eq 1)
;i0 = where(nulls.type eq 0)
in = where(nulls.type eq -1)

p2 = plot(ip, nulls[ip].pos[0]-1,'S r',/overplot,sym_filled=1,sym_size=2)
;p = plot(i0, nulls[i0].pos[0]-1,'S g',/overplot,sym_filled=1,sym_size=2)
p3 = plot(in, nulls[in].pos[0]-1,'S b',/overplot,sym_filled=1,sym_size=2,sym_trans=85)

p['axis2'].delete
p['axis3'].delete
p.xtickdir=1
p.ytickdir=1

p.save, 'nulls.png'

end
