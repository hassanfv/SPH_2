#include <iostream>

using namespace std;

//#define ARRAY_SIZE 10000  // size of the array

int main() {

    
    int nCPUs = 5;
    int N = 127;
    
    int counts[nCPUs];
    int offsets[nCPUs];
    
    
    for (int i = 0; i < nCPUs; i++) 
    {
        counts[i] = N / nCPUs;
        if (i == nCPUs - 1) 
        {
            counts[i] += N % nCPUs;
        }
    offsets[i] = i * (N / nCPUs);
    }   


    for (int i = 0; i < nCPUs; i++)
    {
        cout << counts[i] << ", " << offsets[i] << endl;
    }


    return 0;
}

