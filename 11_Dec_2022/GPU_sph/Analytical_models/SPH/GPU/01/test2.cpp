#include <fstream>
#include <iostream>

int main() {
    // Open the file
    std::ifstream file("params.txt");

    // Check if file was successfully opened
    if (!file) {
        std::cerr << "Unable to open file param.txt";
        return 1; // return with error code 1
    }

    // Variables to store the values
    int N, ND, NG;
    double G;

    // Read the values from the file
    file >> N;
    file >> ND;
    file >> NG;
    file >> G;

    // Close the file
    file.close();

    // Print out the values to check they're correct
    std::cout << "N: " << N << "\n";
    std::cout << "ND: " << ND << "\n";
    std::cout << "NG: " << NG << "\n";
    std::cout << "G: " << G << "\n";

    return 0;
}

