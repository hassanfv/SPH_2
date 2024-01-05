
//*************************************************************************
//*************** Function to save the OUTPUT Snap-Shots!! ****************
//*************************************************************************
void saveArraysToBinary(const std::string &filename, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *rho, float *h, float *u, float *mass, float *ionFrac, int *Typ, int N, int N_ionFrac)
{
  // Open the file in binary mode
  std::ofstream file(filename, std::ios::binary);

  // Check if the file was opened successfully
  if (!file)
  {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  // Write N and NG to the file
  file.write(reinterpret_cast<const char *>(&N), sizeof(int));
  file.write(reinterpret_cast<const char *>(&N_ionFrac), sizeof(int));

  // Write the arrays to the file
  file.write(reinterpret_cast<const char *>(Typ), N * sizeof(int));
  file.write(reinterpret_cast<const char *>(x), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(y), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(z), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(vx), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(vy), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(vz), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(rho), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(h), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(u), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(mass), N * sizeof(float));
  
  file.write(reinterpret_cast<const char *>(ionFrac), N_ionFrac * sizeof(float));

  // Close the file
  file.close();
}










auto T_SaveFile = std::chrono::high_resolution_clock::now();
    //------------ SAVING SNAP-SHOTS ------------
    cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost); // Moved outside so that it can be used by nSplit calculator in ach time step.
    if (!(counter % 200))
    //if (counter > -1)
    {
      cudaMemcpy(Typ, d_Typ, N * sizeof(int), cudaMemcpyDeviceToHost);

      cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(vx, d_vx, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vz, d_vz, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(rho, d_rho, N * sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(u, d_u, N * sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaMemcpy(mass, d_mass, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(ionFrac, d_ionFrac, N_ionFrac * sizeof(float), cudaMemcpyDeviceToHost);

      // Specify the output file name
      std::string filename = "./Outputs/G-" + to_string(t * 10) + ".bin";
      // Save the arrays to binary format
      saveArraysToBinary(filename, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, Typ, N, N_ionFrac);
    }
    auto end_SaveFile = std::chrono::high_resolution_clock::now();
    auto elapsed_SaveFile = std::chrono::duration_cast<std::chrono::nanoseconds>(end_SaveFile - T_SaveFile);
    cout << "T_SaveFile = " << elapsed_SaveFile.count() * 1e-9 << endl;
