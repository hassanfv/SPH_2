pro qsodist

; This program uses the number of photons, ionization parameter, ... to derive the distance from the central AGN.

;+++++++++++ INPUT ++++++++
nH = 180.    ; cm^-3 (Note that this is NOT in log !).
logU = 0.3  ; Log of Ionization parameter (Note that this is the log of U !).
;++++++++++++++++++++++++++

c = 3.D10    ; cm/s (light speed).
U = 10.^logU 
Q = 6.78D55  ;6.78D55  ;photons/sec

r0 = SQRT( Q / (4. * !pi * nH * c * U) )   ; This is in cm.
r0ly = r0 / 9.4605284D17 ; This is now in light years.
r0pc = r0ly / 3.261564

print, '  '
;print, 'The distance in Lightyear is: ', r0ly
print, 'The distance in parsec is: ', r0pc

end
