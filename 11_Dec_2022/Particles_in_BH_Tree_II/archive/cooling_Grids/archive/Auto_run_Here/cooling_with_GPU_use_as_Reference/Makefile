SUNDIALS_PATH = /path/to/install/dir
HDF5_INCL_PATH = /usr/include/hdf5/serial
HDF5_LIB_PATH = /usr/lib/x86_64-linux-gnu/hdf5/serial

CC = gcc 
CFLAGS = -DCHIMES_USE_DOUBLE_PRECISION 
EXEC = libchimes.so

INCL = 
LIBS = -lm 

ifeq ($(strip $(SUNDIALS_PATH)),)
    INCL +=
    LIBS += -lsundials_cvode -lsundials_nvecserial -lsundials_sunlinsoldense -lsundials_sunmatrixdense
else
    INCL += -I$(strip $(SUNDIALS_PATH))/include
    LIBS += -L$(strip $(SUNDIALS_PATH))/lib64 -L$(strip $(SUNDIALS_PATH))/lib -lsundials_cvode -lsundials_nvecserial -lsundials_sunlinsoldense -lsundials_sunmatrixdense
endif 

ifeq ($(strip $(HDF5_INCL_PATH)),)
    INCL += 
else 
    INCL += -I$(strip $(HDF5_INCL_PATH)) 
endif 

ifeq ($(strip $(HDF5_LIB_PATH)),)
    LIBS += -lhdf5 
else 
    LIBS += -L$(strip $(HDF5_LIB_PATH)) -lhdf5 
endif 

OBJS = ./src/chimes_cooling.o ./src/rate_equations.o ./src/update_rates.o ./src/init_chimes.o ./src/chimes.o 

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -shared -o $(EXEC) $(OBJS) $(LIBS) 

./src/chimes.o: ./src/chimes.c ./src/chimes_vars.h ./src/chimes_proto.h ./src/chimes_interpol.h 
	$(CC) $(CFLAGS) -c -Wall -fpic $(INCL) ./src/chimes.c -o ./src/chimes.o

./src/init_chimes.o: ./src/init_chimes.c ./src/chimes_vars.h ./src/chimes_proto.h ./src/chimes_interpol.h 
	$(CC) $(CFLAGS) -c -Wall -fpic $(INCL) ./src/init_chimes.c -o ./src/init_chimes.o

./src/update_rates.o: ./src/update_rates.c ./src/chimes_vars.h ./src/chimes_proto.h ./src/chimes_interpol.h 
	$(CC) $(CFLAGS) -c -Wall -fpic $(INCL) ./src/update_rates.c -o ./src/update_rates.o

./src/rate_equations.o: ./src/rate_equations.c ./src/chimes_vars.h ./src/chimes_proto.h ./src/chimes_interpol.h 
	$(CC) $(CFLAGS) -c -Wall -fpic $(INCL) ./src/rate_equations.c -o ./src/rate_equations.o

./src/chimes_cooling.o: ./src/chimes_cooling.c ./src/chimes_vars.h ./src/chimes_proto.h ./src/chimes_interpol.h 
	$(CC) $(CFLAGS) -c -Wall -fpic $(INCL) ./src/chimes_cooling.c -o ./src/chimes_cooling.o
