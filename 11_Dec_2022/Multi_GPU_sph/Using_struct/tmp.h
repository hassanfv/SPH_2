//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!! START of getDensity !!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for (int i = 0; i < nGPUs; i++)
{
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    getDensity_mgpu_struct<<<gridSize, blockSize>>>(beg[i], end[i], d_p[i], N);
}

// Synchronize to make sure computation is done before proceeding
for (int i = 0; i < nGPUs; i++)
{
    cudaSetDevice(i);
    cudaDeviceSynchronize();
}

// Copy results to main GPU.
int NN = MLen[0];
for (int i = 1; i < nGPUs; i++)
{

    cudaMemcpyPeer(d_p[0] + NN, 0, d_p[i] + NN, i, MLen[i] * sizeof(Particle));

    NN = NN + MLen[i];
}

// The main GPU now has the full results in d_p[0]

// Copy results back to each device for next iteration
for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already has the data!
{

    cudaMemcpyPeer(d_p[i], i, d_p[0], 0, N * sizeof(Particle));
}
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!! END of getDensity !!!!!!!!!!!!!!!!!!!!!!
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!