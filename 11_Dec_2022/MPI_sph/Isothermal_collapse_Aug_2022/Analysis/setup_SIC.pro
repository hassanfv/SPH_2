;+
; NAME: setup_SIC
;
; PURPOSE: 
; Sets up the GADGET initial conditions for the standard isothermal collapse test
;
; CALLING SEQUENCE:
; setup_SIC,nGasPower
;
; INPUTS:
; nGasPower: Sets the number of particles in the simulation.  The
; number of particles will be 2^nGasPower.  Should be an integer.  If
; a non-integer value is supplied, nGasPower is rounded off to the
; nearest integer value.
;
; KEYWORDS:
; regular_grid: Sets up the initial particle distribution to be on a
; regular grid rather than randomly distributed.
;
; OUTPUTS:
; Writes a GADGET initial conditions file in the current working directory.
;
; EXAMPLE:
; setup_SIC,16
;
; MODIFICATION HISTORY:
; Created 6/2/2011 NJG
;-

PRO setup_SIC,nGasPower,regular_grid=regular_grid

;; filename of GADGET IC file to be created
foutbase = "SIC_"                                                     
foutsuffix = ".dat"
fout = foutbase+string(nGasPower,format='(I03)')+foutsuffix

G = 6.6738e-8
msun = 1.98892e33                            


;; Disk properties (directly from Burkert & Bodenheimer 1993, see also Burkert et al. 1997, Springel 2005)
Ngas = 2l^round(nGasPower)                      ;; The number of gas particles in the simulation
mgas = msun                                     ;; The mass of the cloud
rgas = 5d16                                     ;; The initial radius of the cloud in cm
Omega = 7.2d-13                                 ;; The initial angular velocity of the cloud in radians s^-1 
rho0 = 3.82d-18                                 ;; The initial average density
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

; Creating storage variables which will be written to the IC file

pos = fltarr(3,Ngas)
vel = fltarr(3,Ngas)
masses = fltarr(Ngas)
id = lindgen(Ngas)+1
u = fltarr(Ngas)

; Making a uniform grid of particles from which I will carve a sphere
; with Ngas particles.

IF keyword_set(regular_grid) THEN BEGIN
   ninds = max([15,2*round(((1.1*ngas)*3./4/!dpi)^.333333333)])              ;; Size of grid template
   
   inds = findgen(ninds)-ninds/2 
   
   xarr = rebin(inds,ninds,ninds,ninds)                                    ;; 3D grid of particle x positions
   yarr = transpose(xarr,[2,0,1])                                          ;; Same for y positions
   zarr = transpose(xarr,[1,2,0])                                          ;; Same for z positions
   
   gridInds=lindgen(n_elements(xarr))
   
   pos = transpose([[xarr[gridInds]],[yarr[gridInds]],[zarr[gridInds]]])   ;; 3 X ninds^3 array of grid positions
   
   rarr = sqrt(pos[0,*]^2 + pos[1,*]^2 + pos[2,*]^2)                       ;; vector of distance from origin to grid points 
   
   srt = (sort(rarr))[0:ngas-1]                                            ;; Indices for Ngas points closest to origin
   
   IF max(rarr[srt]) GT ninds/2 THEN BEGIN
      print,'Must increase ninds'  ;; You're trying to simulate a cloud with a very large number of particles. 
      stop                         ;; Increase ninds if you want this error to go away.
   ENDIF
   
   pos = pos[*,(sort(rarr))[0:Ngas-1]]                                     ;; Position of particles in sphere
   
   pos *= rgas/max(rarr[srt])                                              ;; Scaling particle positions to where I want them to be in the simulation
   
; Calculating particle velocities in rectangular coordinates
   
   r = sqrt(pos[0,*]^2 + pos[1,*]^2 + pos[2,*]^2)

ENDIF

r = randomu(seed,ngas)*rgas

u = randomu(seed,ngas)*2 - 1
theta = randomu(seed,ngas)*2*!dpi

pos = transpose([[r^(1./3)*sqrt(1-u^2)*cos(theta)],[r^(1./3)*sqrt(1-u^2)*sin(theta)],[r^(1./3)*u]])

rxy = sqrt(pos[0,*]^2 + pos[1,*]^2)

vel[0,*] = -r*omega*pos[1,*]/rxy * rxy/r                                ;; r * omega * cos(theta) * sin(phi)
vel[1,*] = r*omega*pos[0,*]/rxy * rxy/r                                 ;; r * omega * sin(theta) * sin(phi)
vel[2,*] = 0                                                            ;; theta is angle with respec to the z axis, phi is the azimuthal angle

;print, vel

wh=where(finite(vel) NE 1)

IF wh NE -1 THEN BEGIN
   vel[wh] = 0
ENDIF
; Calculating particle masses

mp = Mgas / Ngas

masses = mp * (1 + .1*((pos[0,*]/rxy)^2 - (pos[1,*]/rxy)^2))         ;; Imposing an m=2 density perturbation with an amplitude of 10 percent
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
