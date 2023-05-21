#include <mpi.h>
#include <vector>

using namespace std;

//#define ARRAY_SIZE 10000  // size of the array

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nCPUs;
    MPI_Comm_size(MPI_COMM_WORLD, &nCPUs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    
    int N = 127;
    
    float arr[N];
    float result[N];
    
    
    if (rank == 0)
    {
        
        for (int i = 0; i < N; i++)
        {
            arr[i] = 0.1f * static_cast<float>(i);
        }
    }


    // Distribute the array to all processes
    MPI_Bcast(arr, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
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
    
    cout << endl;
    cout << "rank = " << rank << ", beg = " << beg << ", end = " << end << ", N / nCPUs = " << N / nCPUs << ", end - beg = " << end - beg << endl;
    
    
    // Prepare counts and offsets for gather
    int counts[nCPUs];
    int offsets[nCPUs];
    
    for (int i = 0; i < nCPUs; i++)
    {
    
        if (rank < remainder) 
        {
            beg = rank * (N_per_cpu + 1);
            end = beg + N_per_cpu + 1;
        } else 
        {
            beg = rank * N_per_cpu + remainder;
            end = beg + N_per_cpu;
        }
        
        counts[i] = end - beg;
        offsets[i] = beg;
    }
    
    
    cout << endl;
    for (int i = 0; i < nCPUs; i++)
    {
        cout << "rank = " << rank << ", counts = " << counts[i] << ", offsets = " << offsets[i] << endl;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    MPI_Finalize();

    return 0;
}

