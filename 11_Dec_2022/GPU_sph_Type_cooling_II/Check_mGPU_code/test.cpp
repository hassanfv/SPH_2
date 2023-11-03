#include <iostream>

using namespace std;

int main()
{

  int N = 446865;
  int nGPUs = 8;

  const int N_per_GPU = N / nGPUs;
  const int remainder = N % nGPUs;

  int *beg, *end, *MLen;

  beg = new int[nGPUs];
  end = new int[nGPUs];
  MLen = new int[nGPUs];

  for (int rank = 0; rank < nGPUs; rank++)
  {
      if (rank < remainder)
      {
        beg[rank] = rank * (N_per_GPU + 1);
        end[rank] = beg[rank] + N_per_GPU + 1;
      }
      else
      {
        beg[rank] = rank * N_per_GPU + remainder;
        end[rank] = beg[rank] + N_per_GPU;
      }
      MLen[rank] = end[rank] - beg[rank];
  }


  for (int i = 0; i < nGPUs; i++)
  {
  
    cout << "i = " << i << ", " << "beg = " << beg[i] << ", end = " << end[i] << ", MLen = " << MLen[i] << endl;
  
  }


}
