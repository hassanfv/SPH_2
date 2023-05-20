#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

int main() {

  int N = 33552;
  std::vector<float> xvec(N);
  std::vector<float> yvec(N);
  std::vector<float> zvec(N);
  std::vector<float> vxvec(N);
  std::vector<float> vyvec(N);
  std::vector<float> vzvec(N);
  std::vector<float> massvec(N);
  std::vector<float> hpreviousvec(N);
  std::vector<float> epsvec(N);
  std::vector<float> uvec(N);

  // Check if the binary file exists
  std::ifstream file("IC_AGN_46k.bin", std::ios::binary);
  if (!file) {
    std::cout << "Could not open the binary file." << std::endl;
  }
  file.close();
  
  // Open the binary file for reading
  file.open("IC_AGN_46k.bin", std::ios::binary);
  if (file) {
    // Read the first array
    file.read(reinterpret_cast<char*>(xvec.data()), sizeof(float) * xvec.size());
    file.read(reinterpret_cast<char*>(yvec.data()), sizeof(float) * yvec.size());
    file.read(reinterpret_cast<char*>(zvec.data()), sizeof(float) * zvec.size());
    
    file.read(reinterpret_cast<char*>(vxvec.data()), sizeof(float) * vxvec.size());
    file.read(reinterpret_cast<char*>(vyvec.data()), sizeof(float) * vyvec.size());
    file.read(reinterpret_cast<char*>(vzvec.data()), sizeof(float) * vzvec.size());
    
    file.read(reinterpret_cast<char*>(massvec.data()), sizeof(float) * massvec.size());
    file.read(reinterpret_cast<char*>(hpreviousvec.data()), sizeof(float) * hpreviousvec.size());
    file.read(reinterpret_cast<char*>(epsvec.data()), sizeof(float) * epsvec.size());
    file.read(reinterpret_cast<char*>(uvec.data()), sizeof(float) * uvec.size());

    // Close the file
    file.close();

  } else {
    std::cout << "Failed to open the file." << std::endl;
  }
    
  for (int i = 30000; i < 30010; i++) {
    std::cout << xvec[i] << std::endl;
  }
  
  std::cout << xvec.size() << ", " << epsvec.size() << std::endl;
  
  
  
  
}

