pro map, n, rsphere=rsphere, ball=ball, converge=converge, fan=fan, file=file

  if keyword_set(converge) then begin
    spawn, 'make'
    if file ne !null then spawn, 'sf -i ' + file + ' -n ' + string(n, format='(I4.4)') else spawn, 'sf n=' + string(n, format='(I4.4)')
  endif
  
  plotstruct, n, rsphere=rsphere, ball=ball, converge=converge, fan=fan, file=file
  
end
