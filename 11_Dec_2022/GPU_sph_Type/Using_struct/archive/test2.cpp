#include <iostream>

using namespace std;

int Add(int *a, int *b)
{
  int z = *a + *b;

  return z;
}

int main()
{

  int x = 2, y = 4;
  int z;

  z = Add(&x, &y);

  cout << z << endl;
};