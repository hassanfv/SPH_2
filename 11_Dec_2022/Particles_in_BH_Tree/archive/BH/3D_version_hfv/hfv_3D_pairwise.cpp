//%%writefile test.cu

#include <iostream>
#include <cmath>
#include <random>
#include <fstream>

using namespace std;

const int BLOCK_SIZE = 1024;

#define COLLISION_TH 0.05
#define E 0.1
#define GRAVITY 1.0
#define THETA 0.5

//------ Vector -------
struct Vector
{
  float x;
  float y;
  float z;
};


//------ Body -------
struct Body
{
  bool isDynamic = true;
  float mass;
  float radius;
  Vector position;
  Vector velocity;
  Vector acceleration = {0.0, 0.0, 0.0};
};


// Function to calculate gravitational acceleration on the ith body
void calculateGravitationalAcceleration(Body *bodies, int i, int nBodies) 
{
    Vector totalAcceleration = {0.0, 0.0, 0.0};

    for (int j = 0; j < nBodies; ++j)
    {
        if (i != j)
        {
            Vector rij = {bodies[j].position.x - bodies[i].position.x, bodies[j].position.y - bodies[i].position.y, bodies[j].position.z - bodies[i].position.z};
            float r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z) + (E * E));
            float f = (GRAVITY * bodies[j].mass) / (r * r * r + (E * E));
            Vector acceleration = {rij.x * f, rij.y * f, rij.z * f};

            totalAcceleration.x = totalAcceleration.x + acceleration.x;
            totalAcceleration.y = totalAcceleration.y + acceleration.y;
            totalAcceleration.z = totalAcceleration.z + acceleration.z;
        }
    }

    bodies[i].acceleration = {totalAcceleration.x, totalAcceleration.y, totalAcceleration.z};
}


void readFromFile(const char* filename, Body** h_b, int* nBodies) {
    ifstream file(filename, ios::in | ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file for reading!" << endl;
        return;
    }

    // Read nBodies
    file.read(reinterpret_cast<char*>(nBodies), sizeof(*nBodies));

    // Allocate memory for h_b
    *h_b = new Body[*nBodies];

    // Read bodies
    for (int i = 0; i < *nBodies; ++i) {
        file.read(reinterpret_cast<char*>(&(*h_b)[i]), sizeof(Body));
    }

    file.close();
}



int main()
{

  Body* h_b;
  int nBodies;

  // Read h_b from the binary file
  readFromFile("h_b.bin", &h_b, &nBodies);


  int i = 516000; // Index of the body for which we want to calculate the acceleration
  calculateGravitationalAcceleration(h_b, i, nBodies);

  Body b1 = h_b[i];
  
  float accx = b1.acceleration.x;
  float accy = b1.acceleration.y;
  float accz = b1.acceleration.z;
  cout << "i = " << i << endl;
  printf("(accx, accy, accz) = %f, %f, %f\n", accx, accy, accz);
  cout << "b1.position.x = " << b1.position.x << endl;
  cout << "b1.position.y = " << b1.position.y << endl;
  cout << "b1.position.z = " << b1.position.z << endl;
  cout << endl;
  


}



