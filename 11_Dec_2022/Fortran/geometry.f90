! A module containing functions to compute the area of a circle
! Written by Hassan Fathivavsari, 2022
module geometry
 implicit none
 real, parameter :: pi = 4.*atan(1.)
 public area, pi
 private
 
contains
 ! 
 ! A function to calculate the area of a circle 
 ! 
 real function area(r)
  real, intent(in) :: r
  
  area = pi*r**2
  
 end function area

end module geometry
