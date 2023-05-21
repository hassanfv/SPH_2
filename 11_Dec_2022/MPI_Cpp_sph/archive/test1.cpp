#include <mpi.h>
#include <stdio.h>

#define ARRAY_SIZE 10000  // size of the array

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int data[ARRAY_SIZE];
    int result[ARRAY_SIZE];

    if (world_rank == 0) {
        // Initialize the array
        for (int i = 0; i < ARRAY_SIZE; i++)
            data[i] = i;
    }

    // Distribute the array to all processes
    MPI_Bcast(data, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes its part
    int local_start = world_rank * ARRAY_SIZE / world_size;
    int local_end = (world_rank + 1) * ARRAY_SIZE / world_size;
    for (int i = local_start; i < local_end; i++) {
        result[i] = data[i] * 2;
    }

    // Gather the computed parts to the master process
    MPI_Gather(&result[local_start], local_end - local_start, MPI_INT,
               result, ARRAY_SIZE / world_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Print the result
        for (int i = 0; i < ARRAY_SIZE; i++)
            printf("%d ", result[i]);
        printf("\n");
    }

    MPI_Finalize();

    return 0;
}

