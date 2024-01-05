//--- Finding max_h
  max_h = h[0];
  for (int i = 1; i < N; i++)
  {
    if (h[i] > max_h)
      max_h = h[i];
  }
  
  L_box = max(max(max_x, max_y), max_z);
  
  int nSplit = binSize * L_box / pow(hcoeff, 1.0f/3.0f) / 2.0f / max_h;
  cout << "Adopted nSplit = " << nSplit << endl;
  
  
  
  
  
auto T_nSplit = std::chrono::high_resolution_clock::now();

auto end_nSplit = std::chrono::high_resolution_clock::now();
auto elapsed_nSplit = std::chrono::duration_cast<std::chrono::nanoseconds>(end_nSplit - T_nSplit);
cout << "T_nSplit = " << elapsed_nSplit.count() * 1e-9 << endl;
