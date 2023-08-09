#include <iostream>

using namespace std;

struct Particle
{
  float x, y, z;
};

int main()
{

  const int N = 10;

  Particle *p;

  p = new Particle[N];

  for (int i = 0; i < N; i++)
  {
    p[i].x = i + 0.1f;
    p[i].y = i + 0.2f;
    p[i].z = i + 0.3f;
  }

  cout << p[8].z << endl;

  cout << sizeof(Particle) << endl;
};