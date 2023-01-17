/*
Save all files in the same directory, then
run this test by (gcc= your c compiler, g95= your fortran compiler):

  gcc -c test.c
  g95 -c frt_cf3.F
  g95 frt_cf3.o test.o -o test
  ./test > test.res.c
  diff test.res test.res.c

by Martin.Krause@unige.ch, 2012

 */

#include <stdio.h>
#include <math.h>


void frtinitcf_(const int *mode, const char *path, int len);
void frtcfcache_(const float *Den, const float *Z, 
		 const float *Plw, const float *Ph1, 
		 const float *Pg1, const float *Pc6, 
		 int *icache, float *rcache, int *err);
void frtcfgetln_(const float *alt, int *icache, float *rcache, 
		 float *cfun, float *hfun);
void frtgetcf_(const float *Tem, const float *Den, 
	       const float *Z, const float *Plw, 
	       const float *Ph1, const float *Pg1,
	       const float *Pc6, float *cfun, float *hfun, 
	       int *err);


int mode, ierr;
float Den, Plw, Ph1, Pg1, Pc6, alt, Tem, cfun0, cfun1, hfun0, hfun1;
float Z;

int main(int argc, char *argv[])
{
  mode = 0;
  frtinitcf_(&mode,"cf_table.I2.dat",15);
      if(mode != 0) 
	printf("Error in reading data file: %d \n",mode);
 
      Den = 1.0;
      Plw = 2.11814e-11;
      Ph1 = 1.08928e-7;
      Pg1 = 2.76947e-8;
      Pc6 = 1.03070e-12;

      alt = 1.0;
      while(alt < 9.0) {
	Tem = pow(10.0,alt);
 
	Z = 0.01;
	frtgetcf_(&Tem,&Den,&Z,&Plw,&Ph1,&Pg1,&Pc6,&cfun0,&hfun0,&ierr);
	if(ierr != 0) 
	  printf("Error in table call: %d \n",ierr);

	Z = 0.0;
	frtgetcf_(&Tem,&Den,&Z,&Plw,&Ph1,&Pg1,&Pc6,&cfun1,&hfun1,&ierr);
	if(ierr != 0) 
	  printf("Error in table call: %d \n",ierr);

	printf ("%3.1f %10.3E %10.3E %10.3E %10.3E\n",alt, cfun0, hfun0, cfun1, hfun1);

	alt = alt + 0.1;
      }

}
