import subprocess

# gcc -c test.c
# g95 -c frt_cf3.F
# g95 frt_cf3.o test.o -o test
# ./test > test.res.c

subprocess.call(["gcc -c", "test.c"]) 
#tmp=subprocess.call("./a.out")
