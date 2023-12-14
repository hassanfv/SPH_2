#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

int main() {
    const std::string filename = "h_child.bin";
    
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Determine the size of the file
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of integers
    int num_ints = size / sizeof(int);

    // Read the file into a vector
    std::vector<int> h_child(num_ints);
    if (file.read(reinterpret_cast<char*>(h_child.data()), size)) {
        std::cout << "h_child.shape = " << h_child.size() << std::endl;
    } else {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    // Find the last index where h_child is not -1
    int lastIndex = -1;  // Initialize to -1 to indicate not found
    for (int i = 0; i < num_ints; ++i) {
        if (h_child[i] != -1) {
            lastIndex = i;
        }
    }

    std::cout << "Last index where h_child is not -1: " << lastIndex << std::endl;
    
    if (lastIndex > static_cast<int>(num_ints - 0.1f * num_ints))
      std::cout << "Danger !!!! It seems the h_child (and d_child) size is not enough. Try to increase it !!!" << std::endl;



    return 0;
}

