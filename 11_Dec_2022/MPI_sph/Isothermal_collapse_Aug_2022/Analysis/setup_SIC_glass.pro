;+
; NAME: setup_SIC_glass
;
; PURPOSE: 
; Sets up the GADGET initial conditions for the standard isothermal
; collapse test drawn from glass-like initital conditions provided by Gadget
;
; CALLING SEQUENCE:
; setup_SIC,fname,nGasPower
;
; INPUTS:
; fname: String filename of the final glass-making snapshot 
;
; nGasPower: Sets the number of particles in the simulation.  The
; number of particles will be 2^nGasPower.  Should be an integer.  If
; a non-integer value is supplied, nGasPower is rounded off to the
; nearest integer value.
;
; OUTPUTS:
; Writes a GADGET initial conditions file in the current working directory.
;
; EXAMPLE:
; setup_SIC,fname,16
;
; MODIFICATION HISTORY:
; Created 6/2/2011 NJG
;-

PRO read_snap,fname,ngaspower
  ;; Reading in glass particle positions

  npart=lonarr(6)	
  massarr=dblarr(6)
  time=0.0D
  redshift=0.0D
  flag_sfr=0L
  flag_feedback=0L
  npartTotal=lonarr(6)	
  bytesleft=256-6*4 - 6*8 - 8 - 8 - 2*4-6*4
  la=intarr(bytesleft/2)

  openr,1,fname,/f77_unformatted

  readu,1,npart,massarr,time,redshift,flag_sfr,flag_feedback,npartTotal,la

  N=total(npart)
  pos=fltarr(3,N)
  vel=fltarr(3,N)
  id=lonarr(N)
  
  ind=where((npart gt 0) and (massarr eq 0)) 
  if ind(0) ne -1 then begin
     Nwithmass= total(npart(ind))
     mass=fltarr(Nwithmass)
  endif else begin	
     Nwithmass= 0
  endelse

  readu,1,pos
  readu,1,vel
  readu,1,id

  if Nwithmass gt 0 then begin
     readu,1,mass
  endif

  NGas=npart(0)

  if Ngas gt 0 then begin

     u=fltarr(Ngas)
     readu,1,u

     rho=fltarr(Ngas)
     readu,1,rho
  endif
  close,1
  
  ;; Generating a much bigger box by mirroring the box in the snapshot
  ;; file to produce cube with 8 times more volume.
  
  boxsize = 50000
  xpos = reform(pos[0,*])
  ypos = reform(pos[1,*])
  zpos = reform(pos[2,*])

  newxpos = [xpos,xpos,xpos-boxsize,xpos-boxsize,xpos,xpos,xpos-boxsize,xpos-boxsize]
  newypos = [ypos,ypos-boxsize,ypos-boxsize,ypos,ypos,ypos-boxsize,ypos-boxsize,ypos]
  newzpos = [zpos,zpos,zpos,zpos,zpos-boxsize,zpos-boxsize,zpos-boxsize,zpos-boxsize]

  pos = transpose([[newxpos],[newypos],[newzpos]])+boxsize

  boxsize = 100000
  xpos = reform(pos[0,*])
  ypos = reform(pos[1,*])
  zpos = reform(pos[2,*])

  newxpos = [xpos,xpos,xpos-boxsize,xpos-boxsize,xpos,xpos,xpos-boxsize,xpos-boxsize]
  newypos = [ypos,ypos-boxsize,ypos-boxsize,ypos,ypos,ypos-boxsize,ypos-boxsize,ypos]
  newzpos = [zpos,zpos,zpos,zpos,zpos-boxsize,zpos-boxsize,zpos-boxsize,zpos-boxsize]

  pos = transpose([[newxpos],[newypos],[newzpos]])

  ;; filename of GADGET IC file to be created
  foutbase = "SIC_"                                                     
  foutsuffix = ".dat"
  fout = foutbase+string(nGasPower,format='(I03)')+foutsuffix
  
  G = 6.6738e-8
  msun = 1.98892e33                            
  

;; Disk properties (directly from BB 93, see also Burkert et al. 1997, Springel 2005)
  Ngas = 2l^round(nGasPower)                      ;; The number of gas particles in the simulation
  mgas = msun                                     ;; The mass of the cloud
  rgas = 5d16                                     ;; The initial radius of the cloud in cm
  Omega = 7.2d-13                                 ;; The initial angular velocity of the cloud in radians s^-1 
  rho0 = 3.82d-18                                 ;; The initial average gas density
  cs = 1.66e4                                     ;; The sound speed
  
;;  Calculating derived quantities
  tff = sqrt(3*!pi/(32*G*rho0))                   ;; The free-fall time = 3.4e4 yr
  
;; Setting the units of the simulation
  unitMass_in_g = Msun                                                 ;; 1 solar mass
  unitTime_in_s = tff                                                  ;; Scaling time to free-fall time
  unitLength_in_cm = rgas                                              ;; Scaling distance to the initial cloud radius
  unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s          ;; The internal velocity unit
  
; Scaling things to code units
  
  rgas /= UnitLength_in_cm
  mgas /= UnitMass_in_g
  omega *= UnitTime_in_s
  
  vel = fltarr(3,Ngas)
  masses = fltarr(Ngas)
  id = lindgen(Ngas)+1
  u = fltarr(Ngas)

  ngas = 2l^ngaspower
  
  r = sqrt(pos[0,*]^2 + pos[1,*]^2 + pos[2,*]^2)
  
  srt = (sort(r))[0:ngas-1]

  pos = pos[*,srt]                                     ;; Position of particles in sphere
  
  pos *= rgas/max(r[srt]) 

  rxy = sqrt(pos[0,*]^2 + pos[1,*]^2)
  
  vel[0,*] = -r*omega*pos[1,*]/rxy * rxy/r                 ;; r * omega * cos(theta) * sin(phi)
  vel[1,*] = r*omega*pos[0,*]/rxy * rxy/r                  ;; r * omega * sin(theta) * sin(phi)
  vel[2,*] = 0                                             ;; theta is angle with respec to the z axis, phi is the azimuthal angle
  
  wh=where(finite(vel) NE 1)
  IF wh NE -1 THEN BEGIN
     vel[wh] = 0
  ENDIF
; Calculating particle masses
  
  mp = Mgas / Ngas
  
  masses = reform(mp * (1 + .1*((pos[0,*]/rxy)^2 - (pos[1,*]/rxy)^2))) ;; Imposing an m=2 density perturbation with an amplitude of 10 percent
  boxsize = rgas/unitlength_in_cm
  
  
  wh=where(finite(masses) NE 1)
  
  IF (wh NE [-1]) THEN masses[wh] = mp                                 ;; Fixes an issue with the particle at the origin
  
; Assign particle sound speed (isothermal EOS)
  
  u[*] = cs^2 / unitVelocity_in_cm_per_s^2
  
; create header
  
  npart=lonarr(6)
  massarr=dblarr(6)
  time=0.0D
  redshift=0.0D
  flag_sfr=0L
  flag_feedback=0L
  npartTotal=lonarr(6)
  flag_cooling=0L
  num_files = 1L
  Omega0=0.0D
  OmegaLambda=0.0D
  HubbleParam=0.0D
  flag_stellarage=0L
  flag_metals=0L
  highwork=lonarr(6)
  flag_entropy=0L
  flag_double=0L
  flag_lpt=0L
  factor=0.0
  la=intarr(48/2)
  
  
  npart[0] = Ngas
  nparttotal[0] = Ngas
  massarr[0] = 0

; write file

  pos = float(pos)
  vel = float(vel)
  id  = long(id)
  masses = float(masses)
  u = float(u)

  openw, 1, fout, /f77_unformatted

  writeu,1,npart,massarr,time,redshift,flag_sfr,flag_feedback,npartTotal, $
         flag_cooling, num_files, BoxSize, Omega0, OmegaLambda, HubbleParam, flag_stellarage, flag_metals, highwork,$
         flag_entropy, flag_double, flag_lpt, factor, la

  writeu,1,pos
  writeu,1,vel
  writeu,1,id
  writeu,1,masses
  writeu,1,u

  close,1

END
   








