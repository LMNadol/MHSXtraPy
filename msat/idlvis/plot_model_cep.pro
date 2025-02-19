pro plot_model_cep,oModel,theta=theta,phi=phi,psi=psi,scale=scale,shift=shift, $
  dimensions=dimensions,filename=filename,indexed=indexed

 ; oModel=create_magnetosphere_model(10)
  oView=obj_new("IDLgrView")
  oView->add,oModel
  oView->setProperty,zclip=[-12,12],eye=15
  oModel->reset
  oModel->translate,0.,0.,0.
  oModel->scale,1./5.2,1./5.2,1./5.2
  ;oModel->rotate,[1,0,0],-90
  ;oModel->rotate,[0,1,0],90
  
  if keyword_set(theta) then oModel->rotate,[0,1,0],theta
  if n_elements(scale) eq 1 then oModel->scale,scale,scale,scale
  if keyword_set(phi) then oModel->rotate,[1,0,0],phi
  if keyword_set(psi) then oModel->rotate,[0,0,1],psi
  if n_elements(shift) eq 2 then oModel->translate,shift[0],shift[1],0
  if n_elements(shift) eq 3 then oModel->translate,shift[0],shift[1],shift[2]
  
  ;oModel->scale,.7,.7,.7

  ; adjust for imperfect polygons
  if n_elements(dimensions) lt 2 then dimensions=[500,500]
  ratio=dimensions[1]/double(dimensions[0])
  oModel->scale,sqrt(ratio),1/sqrt(ratio),1.0
  
  oBuffer=obj_new("IDLgrBuffer",dimensions=dimensions)
  oBuffer->draw,oView
  oImage=oBuffer->read()
  oImage->getProperty,data=im
  if keyword_set(indexed) then begin
    im=color_quan(im,1,rr,gg,bb,colors=(indexed gt 1)? indexed : 256)
    if keyword_set(filename) then write_png,filename,im,rr,gg,bb
    tvlct,rr_old,gg_old,bb_old,/get & tvlct,rr,gg,bb
    tv,im
    tvlct,rr_old,gg_old,bb_old
  endif else begin
    if keyword_set(filename) then write_png,filename,im
    tv,im,true=1  
  endelse
  oView->remove,oModel
  obj_destroy,[oView,oBuffer,oImage]

end
