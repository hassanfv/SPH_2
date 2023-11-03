#include <iostream>

using namespace std;


void increment(int *p)
{

  *p = *p + 1;

}


int main()
{

  int a = 5;
  
  increment(&a);
  
  cout << "a = " << a << endl;

}
