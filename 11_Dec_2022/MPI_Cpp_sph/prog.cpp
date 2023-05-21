#include <mpi.h>
#include <vector>

using namespace std;

//#define ARRAY_SIZE 10000  // size of the array


void multiply_elements(int start, int end, float *src, float *dst) 
{
    for (int i = start; i < end; i++) 
    {
        dst[i] = src[i] * 0.5f;
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nCPUs;
    MPI_Comm_size(MPI_COMM_WORLD, &nCPUs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    
    int N = 127;
    
    float arr[N];
    float result[N];
    
    
    int N_per_cpu = N / nCPUs;
    int remainder = N % nCPUs;

    int beg, end;
    if (rank < remainder) 
    {
        beg = rank * (N_per_cpu + 1);
        end = beg + N_per_cpu + 1;
    } else 
    {
        beg = rank * N_per_cpu + remainder;
        end = beg + N_per_cpu;
    }
    
    
    if (rank == 0)
    {
        
        for (int i = 0; i < N; i++)
        {
            arr[i] = 0.1f * static_cast<float>(i);
        }
    }


    // Distribute the array to all processes
    MPI_Bcast(arr, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    
    // Call the multiplication function
    multiply_elements(beg, end, arr, result);


    // Prepare counts and offsets for gather
    int counts[nCPUs];
    int offsets[nCPUs];
    
    int begx, endx;
    
    for (int i = 0; i < nCPUs; i++)
    {
    
        if (i < remainder) 
        {
            begx = i * (N_per_cpu + 1);
            endx = begx + N_per_cpu + 1;
        } else 
        {
            begx = i * N_per_cpu + remainder;
            endx = begx + N_per_cpu;
        }
        
        counts[i] = endx - begx;
        offsets[i] = begx;
    }

    
    // Gather the computed parts to the master process
    MPI_Gatherv(&result[beg], end - beg, MPI_FLOAT,
                result, counts, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
        
            cout << result[i] << endl;
        
        }
    
    }


    MPI_Finalize();

    return 0;
}

