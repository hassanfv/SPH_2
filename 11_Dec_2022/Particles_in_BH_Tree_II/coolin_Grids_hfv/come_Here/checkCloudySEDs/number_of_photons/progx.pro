pro progx

compile_opt defint32

h  = 6.6261D-27  ;# Planck constant [erg.s]
c  = 3.0D+18     ;# Speed of light  [Angstrom/s]
hc = h * c

L_lamb912 = 3.25D+42  ;# erg/s/A

lamb0 = 45.  ;# Angstrom [i.e. 20 Ryd]
lamb1 = 912. ;# Angstrom [i.e. 1 Ryd]

wgrid = stepvect(lamb0, lamb1, 0.01)
n = n_elements(wgrid)

dlamb = wgrid[1] - wgrid[0]
L_lamb = fltarr(n)
L_lamb = L_lamb912 + L_lamb  ;# Constant Luminosity along the wavelength.

s = 0.
for i = 0 , n - 2 do begin
s = s + (dlamb * wgrid[i] * L_lamb[i])/hc
endfor

print, s


end
