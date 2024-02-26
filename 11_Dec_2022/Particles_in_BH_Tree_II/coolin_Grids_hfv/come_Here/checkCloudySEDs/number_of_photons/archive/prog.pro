pro prog

compile_opt defint32

lam = stepvect(45., 912., 0.01)
n = n_elements(lam)
Lhlam = dblarr(n)

h = 6.626D-27   ;in erg.s
c = 3.D18  ; The speed of light in Angstrom.

L912 = 1.9895D+42       ;erg/s/A
L    = L912 + 0.*lam

for i = 0 , n - 1 do begin
Lhlam[i] = L(i) / ((h * c)/lam(i)) ;# This is 'L / hnu' which is the number of photons at each frequency or wavelength.
endfor
cgplot, lam, Lhlam

;# So now we have both 'lamb' and 'L/hnu'. We need these two to calculate the TOTAL number of photons from 45 A to 912 A!
;# NOTE that it is not just calculating the area of a simple rectangular. Note that Lhlam is a function of wavelength !!! So INTEGRATE it !!!

s = 0.
for i = 0 , n - 2 do begin
s = s + (lam(i+1) - lam(i)) * Lhlam(i)
endfor

print, 'Number of photons with energies between 1 to 20 rydberg is ', s

end
