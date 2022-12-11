! module to define the precision of real variables.
module precision
 implicit none
 integer, parameter :: dp = selected_real_kind(P=10,R=30)
 integer, parameter :: sp = selected_real_kind(P=5,R=15)
 integer, parameter :: dp_alt = kind(0.d0)
 public dp, sp, dp_alt, print_kind_info

 private
 
contains
 ! to print information about precision
 subroutine print_kind_info()
 real(sp) :: pi_single
 real(dp) :: pi_double
 
 pi_single = 4._sp * atan(1.0_sp)
 pi_double = 4._dp * atan(1.0_dp)
 
 print*, ' sing prec is kind= ', sp
 print*, ' doub prec is kind= ', dp
 print*, ' kind of a double precision number is ', dp_alt
 
 print*, ' pi in single is ', pi_single
 print*, ' pi in double is ', pi_double
 
 
 end subroutine print_kind_info

 
end module precision
