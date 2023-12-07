auto T_MovingData_8 = std::chrono::high_resolution_clock::now();


auto end_MovingData_8 = std::chrono::high_resolution_clock::now();
auto elapsed_MovingData_8 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_MovingData_8 - T_MovingData_8);
cout << "T_MovingData_8 = " << elapsed_MovingData_8.count() * 1e-9 << endl;
