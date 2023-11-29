




__global__ void ngbDB_v3(int *Typ, float *x, float *y, float *z, float *h, int *ngb, int MAX_ngb, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // Declare shared memory array to store neighbor indices for each thread.
  // The size needs to be determined based on your hardware and problem requirements.
  // If MAX_ngb is large, you might need a different approach as shared memory is limited.
  extern __shared__ int s_ngb[];

  if ((i < N) && (Typ[i] == 0))
  {
    float coeff = 0.05f;
    float counter = 0.0f; 
    
    float h_new = 0.0f;

    int k = 0;
    int j = 0;

    float dx, dy, dz;
    float rr;

    while (k < 65)
    {
      k = 0;
      h_new = (2.0f + (counter + 1.0f) * coeff) * h[i];
      j = 0;
      
      // Use a thread-specific offset for each thread in the block.
      int s_index = threadIdx.x * MAX_ngb;
      
      while (j < N && k < MAX_ngb)
      {
        if (Typ[j] == 0)
        {
          dx = x[j] - x[i];
          dy = y[j] - y[i];
          dz = z[j] - z[i];
          rr = sqrt(dx * dx + dy * dy + dz * dz);

          if (rr <= h_new)
          {
            // Store the neighbor index in shared memory.
            s_ngb[s_index + k] = j;
            k++;
          }
        }
        j++;
      }
      counter++;
    }

    __syncthreads();

    // After finding all neighbors, write the results from shared memory to global memory.
    if (threadIdx.x == 0) // We use the thread with Idx.x = 0 to do the transfer from the shared memory to the global memory!
    {
      for (int n = 0; n < k; ++n)
      {
        ngb[i * MAX_ngb + n] = s_ngb[threadIdx.x * MAX_ngb + n];
      }
    }
  }
}





