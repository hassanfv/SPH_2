#include <iostream>

using namespace std;


int SumOfElements(int *A, int N)
{

  int s = 0;
  for (int i = 0; i < N; i++)
  {
    //s += A[i];
    s += *(A + i);
  }

  return s;

}


int main()
{

  int A[] = {1, 2, 3, 4, 5};
  
  int s = SumOfElements(A, 5);
  
  cout << "sum = " << s << endl;

}
