


//======================================== Updated: 19 Nov 2023.
//========== Smoothing Length ============ Updated: 28 Jan 2023. h_new adopted from eq.31 in Gadget2 Paper
//========================================
__global__ void smoothing_h_ngb_new(int *Typ, float *x, float *y, float *z, float *h,
                                    float x_min, float y_min, float z_min, float W_cell, int nSplit,
                                    int *offSet, int *groupedIndex,
                                    int N, int Ndown, int Nup, float coeff,
                                    float Nngb_f, float *Nngb_previous, float *divV, float dt)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float h_new = 2.0f * (0.5f * h[i] * (1.0f + pow((Nngb_f / Nngb_previous[i]), 1.0f / 3.0f)) +
                          1.0f / 3.0f * h[i] * divV[i] * dt);
    
    
    //--------- Finding the neighboring cells -------------
    int neighbors[27]; // highly likely will stay in the register of the thread!
    
    int cell_id = getCelliD(x[i], y[i], z[i], x_min, y_min, z_min, W_cell, nSplit);
    
    // Convert linear index (i.e. cell_id) to 3D coordinates
    int zcell = cell_id / (nSplit * nSplit);
    int ycell = (cell_id / nSplit) % nSplit;
    int xcell = cell_id % nSplit;
    
    int count = 0;  // Count of valid neighbors

    // Loop through all neighboring coordinates
    for (int dz = -1; dz <= 1; dz++)
    {
      for (int dy = -1; dy <= 1; dy++)
      {
        for (int dx = -1; dx <= 1; dx++)
        {
          int nx = xcell + dx;
          int ny = ycell + dy;
          int nz = zcell + dz;

          // Check if the neighbor is within bounds
          if (nx >= 0 && nx < nSplit && ny >= 0 && ny < nSplit && nz >= 0 && nz < nSplit)
          {
            // Convert 3D coordinates back to linear index (i.e. cell_id) and add to neighbors
            neighbors[count++] = nz * nSplit * nSplit + ny * nSplit + nx;
          }
        }
      }
    }
    //-----------------------------------------------------

    int N_iter = 0;
    int k = 0;
    int j = 0;
    
    int k_pre = Nup; // Just a choice so that it does not activate that if condition in the first run!

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {
      k = 0;

      for (int nj = 0; nj < count; nj++)
      {
        int starting_nx = offSet[neighbors[nj]];
        int ending_nx = offSet[neighbors[nj] + 1];
        
        int N_tmp = ending_nx - starting_nx;

        for (int jj = starting_nx; jj < N_tmp; jj++)
        {
          j = groupedIndex[jj];
        
          if (Typ[j] == 0)
          {
            dx = x[j] - x[i];
            dy = y[j] - y[i];
            dz = z[j] - z[i];
            float rr = sqrt(dx * dx + dy * dy + dz * dz);

            if (rr <= h_new)
            {
              k++;
            }
          }
          
          //-----------
          if (k > Nup) // To stop unnecessary search after it reaches k > Nup!
          {
            break;
          }
          //-----------
        }
      }
      

      //-----------
      if (((k < Ndown) && (k_pre > Nup)) || ((k > Nup) && (k_pre < Ndown))) // To prevent oscillation outside Nup and Ndown values!!
      {
        coeff = coeff / 2.0f;
      }
      //-----------

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * h[i];
      }
      
      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * h[i];
      }
      
      k_pre = k;

      N_iter++;
      if (N_iter > 500)
      {
        break;
      }
    }
    Nngb_previous[i] = k;
    h[i] = 0.5 * h_new;
  }
}


