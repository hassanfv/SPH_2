auto T_assess_1 = std::chrono::high_resolution_clock::now();

auto end_assess_1 = std::chrono::high_resolution_clock::now();
auto elapsed_assess_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_assess_1 - T_assess_1);
cout << "T_assess_1 = " << elapsed_assess_1.count() * 1e-9 << endl;
