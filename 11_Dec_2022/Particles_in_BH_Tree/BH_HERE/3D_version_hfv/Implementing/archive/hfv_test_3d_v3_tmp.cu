%%writefile test.cu

#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>

using namespace std;

const int BLOCK_SIZE = 256;

#define COLLISION_TH 0.05
#define E 0.1
#define GRAVITY 1.0
#define THETA 0.8

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
  Vector acceleration;
};


//------ Node -------
struct Node
{
  Vector minCorner; // Minimum corner of the bounding box
  Vector maxCorner; // Maximum corner of the bounding box
  Vector centerMass;
  float totalMass;
  bool isLeaf;
  int start;
  int end;
};


/*
----------------------------------------------------------------------------------------
RESET KERNEL
----------------------------------------------------------------------------------------
*/
__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < nNodes)
    {
        node[b].minCorner = {INFINITY, INFINITY, INFINITY};
        node[b].maxCorner = {-INFINITY, -INFINITY, -INFINITY};
        node[b].centerMass = {-1, -1, -1};
        node[b].totalMass = 0.0;
        node[b].isLeaf = true;
        node[b].start = -1;
        node[b].end = -1;
        mutex[b] = 0;
    }

    if (b == 0)
    {
        node[b].start = 0;
        node[b].end = nBodies - 1;
    }
}



/*
----------------------------------------------------------------------------------------
COMPUTE BOUNDING BOX
----------------------------------------------------------------------------------------
*/
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies)
{
    __shared__ float minX[BLOCK_SIZE];
    __shared__ float minY[BLOCK_SIZE];
    __shared__ float minZ[BLOCK_SIZE];
    __shared__ float maxX[BLOCK_SIZE];
    __shared__ float maxY[BLOCK_SIZE];
    __shared__ float maxZ[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = tx + blockIdx.x * blockDim.x;

    minX[tx] = INFINITY;
    minY[tx] = INFINITY;
    minZ[tx] = INFINITY;
    maxX[tx] = -INFINITY;
    maxY[tx] = -INFINITY;
    maxZ[tx] = -INFINITY;

    __syncthreads();

    if (b < nBodies)
    {
        Body body = bodies[b];
        minX[tx] = body.position.x;
        minY[tx] = body.position.y;
        minZ[tx] = body.position.z;
        maxX[tx] = body.position.x;
        maxY[tx] = body.position.y;
        maxZ[tx] = body.position.z;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            minX[tx] = fminf(minX[tx], minX[tx + s]);
            minY[tx] = fminf(minY[tx], minY[tx + s]);
            minZ[tx] = fminf(minZ[tx], minZ[tx + s]);
            maxX[tx] = fmaxf(maxX[tx], maxX[tx + s]);
            maxY[tx] = fmaxf(maxY[tx], maxY[tx + s]);
            maxZ[tx] = fmaxf(maxZ[tx], maxZ[tx + s]);
        }
    }

    if (tx == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0)
            ;
        node[0].minCorner.x = fminf(node[0].minCorner.x, minX[0] - 0.01);
        node[0].minCorner.y = fminf(node[0].minCorner.y, minY[0] - 0.01);
        node[0].minCorner.z = fminf(node[0].minCorner.z, minZ[0] - 0.01);
        node[0].maxCorner.x = fmaxf(node[0].maxCorner.x, maxX[0] + 0.01);
        node[0].maxCorner.y = fmaxf(node[0].maxCorner.y, maxY[0] + 0.01);
        node[0].maxCorner.z = fmaxf(node[0].maxCorner.z, maxZ[0] + 0.01);
        atomicExch(mutex, 0);
    }
}




/*
----------------------------------------------------------------------------------------
CONSTRUCT QUAD TREE
----------------------------------------------------------------------------------------
*/
__device__ int getOctant(Vector minCorner, Vector maxCorner, float x, float y, float z)
{
    // Calculate the center of the bounding box
    float centerX = (minCorner.x + maxCorner.x) / 2;
    float centerY = (minCorner.y + maxCorner.y) / 2;
    float centerZ = (minCorner.z + maxCorner.z) / 2;

    int octant = 0;

    // Determine the octant
    if (x <= centerX) {
        if (y <= centerY) {
            if (z <= centerZ) {
                octant = 1; // Lower-left-back
            } else {
                octant = 2; // Lower-left-front
            }
        } else {
            if (z <= centerZ) {
                octant = 3; // Upper-left-back
            } else {
                octant = 4; // Upper-left-front
            }
        }
    } else {
        if (y <= centerY) {
            if (z <= centerZ) {
                octant = 5; // Lower-right-back
            } else {
                octant = 6; // Lower-right-front
            }
        } else {
            if (z <= centerZ) {
                octant = 7; // Upper-right-back
            } else {
                octant = 8; // Upper-right-front
            }
        }
    }

    return octant;
}



__device__ void UpdateChildBound(Vector &minCorner, Vector &maxCorner, Node &childNode, int octant)
{
    // Calculate the center of the bounding box
    float centerX = (minCorner.x + maxCorner.x) / 2;
    float centerY = (minCorner.y + maxCorner.y) / 2;
    float centerZ = (minCorner.z + maxCorner.z) / 2;

    // Update the bounds based on the octant
    if (octant == 1) { // Lower-left-back
        childNode.minCorner = {minCorner.x, minCorner.y, minCorner.z};
        childNode.maxCorner = {centerX, centerY, centerZ};
    } else if (octant == 2) { // Lower-left-front
        childNode.minCorner = {minCorner.x, minCorner.y, centerZ};
        childNode.maxCorner = {centerX, centerY, maxCorner.z};
    } else if (octant == 3) { // Upper-left-back
        childNode.minCorner = {minCorner.x, centerY, minCorner.z};
        childNode.maxCorner = {centerX, maxCorner.y, centerZ};
    } else if (octant == 4) { // Upper-left-front
        childNode.minCorner = {minCorner.x, centerY, centerZ};
        childNode.maxCorner = {centerX, maxCorner.y, maxCorner.z};
    } else if (octant == 5) { // Lower-right-back
        childNode.minCorner = {centerX, minCorner.y, minCorner.z};
        childNode.maxCorner = {maxCorner.x, centerY, centerZ};
    } else if (octant == 6) { // Lower-right-front
        childNode.minCorner = {centerX, minCorner.y, centerZ};
        childNode.maxCorner = {maxCorner.x, centerY, maxCorner.z};
    } else if (octant == 7) { // Upper-right-back
        childNode.minCorner = {centerX, centerY, minCorner.z};
        childNode.maxCorner = {maxCorner.x, maxCorner.y, centerZ};
    } else if (octant == 8) { // Upper-right-front
        childNode.minCorner = {centerX, centerY, centerZ};
        childNode.maxCorner = {maxCorner.x, maxCorner.y, maxCorner.z};
    }
}


__device__ void warpReduce(volatile float *totalMass, volatile float3 *centerMass, int tx)
{
    totalMass[tx] += totalMass[tx + 32];
    centerMass[tx].x += centerMass[tx + 32].x;
    centerMass[tx].y += centerMass[tx + 32].y;
    centerMass[tx].z += centerMass[tx + 32].z;

    totalMass[tx] += totalMass[tx + 16];
    centerMass[tx].x += centerMass[tx + 16].x;
    centerMass[tx].y += centerMass[tx + 16].y;
    centerMass[tx].z += centerMass[tx + 16].z;

    totalMass[tx] += totalMass[tx + 8];
    centerMass[tx].x += centerMass[tx + 8].x;
    centerMass[tx].y += centerMass[tx + 8].y;
    centerMass[tx].z += centerMass[tx + 8].z;

    totalMass[tx] += totalMass[tx + 4];
    centerMass[tx].x += centerMass[tx + 4].x;
    centerMass[tx].y += centerMass[tx + 4].y;
    centerMass[tx].z += centerMass[tx + 4].z;

    totalMass[tx] += totalMass[tx + 2];
    centerMass[tx].x += centerMass[tx + 2].x;
    centerMass[tx].y += centerMass[tx + 2].y;
    centerMass[tx].z += centerMass[tx + 2].z;

    totalMass[tx] += totalMass[tx + 1];
    centerMass[tx].x += centerMass[tx + 1].x;
    centerMass[tx].y += centerMass[tx + 1].y;
    centerMass[tx].z += centerMass[tx + 1].z;
}



__device__ void ComputeCenterMass(Node &curNode, Body *bodies, float *totalMass, float3 *centerMass, int start, int end)
{
    int tx = threadIdx.x;
    int total = end - start + 1;
    int sz = ceil((float)total / blockDim.x);
    int s = tx * sz + start;
    float M = 0.0;
    float3 R = make_float3(0.0, 0.0, 0.0);

    for (int i = s; i < s + sz; ++i)
    {
        if (i <= end)
        {
            Body &body = bodies[i];
            M += body.mass;
            R.x += body.mass * body.position.x;
            R.y += body.mass * body.position.y;
            R.z += body.mass * body.position.z;
        }
    }

    totalMass[tx] = M;
    centerMass[tx] = R;

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        __syncthreads();
        if (tx < stride)
        {
            totalMass[tx] += totalMass[tx + stride];
            centerMass[tx].x += centerMass[tx + stride].x;
            centerMass[tx].y += centerMass[tx + stride].y;
            centerMass[tx].z += centerMass[tx + stride].z;
        }
    }

    if (tx < 32)
    {
        warpReduce(totalMass, centerMass, tx);
    }
    __syncthreads();

    if (tx == 0)
    {
        centerMass[0].x /= totalMass[0];
        centerMass[0].y /= totalMass[0];
        centerMass[0].z /= totalMass[0];
        curNode.totalMass = totalMass[0];
        curNode.centerMass = {centerMass[0].x, centerMass[0].y, centerMass[0].z};
    }
}




__device__ void CountBodies(Body *bodies, Vector minCorner, Vector maxCorner, int *count, int start, int end, int nBodies)
{
    int tx = threadIdx.x;
    if (tx < 8) // There are 8 octants in 3D space
        count[tx] = 0;
    __syncthreads();

    for (int i = start + tx; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int octant = getOctant(minCorner, maxCorner, body.position.x, body.position.y, body.position.z);
        atomicAdd(&count[octant - 1], 1);
    }

    __syncthreads();
}




__device__ void ComputeOffset(int *count, int start)
{
    int tx = threadIdx.x;
    if (tx < 8) // For 8 octants in 3D space
    {
        int offset = start;
        for (int i = 0; i < tx; ++i)
        {
            offset += count[i];
        }
        count[tx + 8] = offset; // Store the offset for each octant
    }
    __syncthreads();
}




__device__ void GroupBodies(Body *bodies, Body *buffer, Vector minCorner, Vector maxCorner, int *count, int start, int end, int nBodies)
{
    int *offsets = &count[8]; // Updated for 8 octants
    for (int i = start + threadIdx.x; i <= end; i += blockDim.x)
    {
        //if (i < nBodies)
        //{
            Body body = bodies[i];
            int octant = getOctant(minCorner, maxCorner, body.position.x, body.position.y, body.position.z) - 1;
            int dest = atomicAdd(&offsets[octant], 1);
            buffer[dest] = body;
        //}
    }
    __syncthreads();
}




__global__ void ConstructQuadTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit)
{
    __shared__ int count[16];
    __shared__ float totalMass[BLOCK_SIZE];
    __shared__ float3 centerMass[BLOCK_SIZE];
    int tx = threadIdx.x;
    nodeIndex += blockIdx.x;

    if (nodeIndex > 990000)
      printf("nodeIndex, blockIdx.x = %d, %d\n", nodeIndex, blockIdx.x);

    if (nodeIndex >= nNodes)
        return;

    Node &curNode = node[nodeIndex];
    int start = curNode.start;
    int end = curNode.end;
    
    Vector minCorner = curNode.minCorner;
    Vector maxCorner = curNode.maxCorner;

    if (start == -1 && end == -1)
        return;
    
    if (end - start < 50) //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      return;

    ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);
    if (nodeIndex >= leafLimit || start == end)
    {
        for (int i = start; i <= end; ++i)
        {
            buffer[i] = bodies[i];
        }

        return;
    }

    CountBodies(bodies, minCorner, maxCorner, count, start, end, nBodies);
    ComputeOffset(count, start);
    GroupBodies(bodies, buffer, minCorner, maxCorner, count, start, end, nBodies);

    if (tx == 0)
    {
        Node &LLBNode = node[(nodeIndex * 8) + 1];
        Node &LLFNode = node[(nodeIndex * 8) + 2];
        Node &ULBNode = node[(nodeIndex * 8) + 3];
        Node &ULFNode = node[(nodeIndex * 8) + 4];
        Node &LRBNode = node[(nodeIndex * 8) + 5];
        Node &LRFNode = node[(nodeIndex * 8) + 6];
        Node &URBNode = node[(nodeIndex * 8) + 7];
        Node &URFNode = node[(nodeIndex * 8) + 8];
    
        UpdateChildBound(minCorner, maxCorner, LLBNode, 1);
        UpdateChildBound(minCorner, maxCorner, LLFNode, 2);
        UpdateChildBound(minCorner, maxCorner, ULBNode, 3);
        UpdateChildBound(minCorner, maxCorner, ULFNode, 4);
        UpdateChildBound(minCorner, maxCorner, LRBNode, 5);
        UpdateChildBound(minCorner, maxCorner, LRFNode, 6);
        UpdateChildBound(minCorner, maxCorner, URBNode, 7);
        UpdateChildBound(minCorner, maxCorner, URFNode, 8);

        curNode.isLeaf = false;
        
        if (count[0] > 0)
        {
            LLBNode.start = start;
            LLBNode.end = start + count[0] - 1;
        }
        
        if (count[1] > 0)
        {
            LLFNode.start = start + count[0];
            LLFNode.end = start + count[0] + count[1] - 1;
        }
        
        if (count[2] > 0)
        {
            ULBNode.start = start + count[0] + count[1];
            ULBNode.end = start + count[0] + count[1] + count[2] - 1;
        }
        
        if (count[3] > 0)
        {
            ULFNode.start = start + count[0] + count[1] + count[2];
            ULFNode.end = start + count[0] + count[1] + count[2] + count[3] - 1;
        }
        
        if (count[4] > 0)
        {
            LRBNode.start = start + count[0] + count[1] + count[2] + count[3];
            LRBNode.end = start + count[0] + count[1] + count[2] + count[3] + count[4] - 1;
        }
        
        if (count[5] > 0)
        {
            LRFNode.start = start + count[0] + count[1] + count[2] + count[3] + count[4];
            LRFNode.end = start + count[0] + count[1] + count[2] + count[3] + count[4] + count[5] - 1;
        }
        
        if (count[6] > 0)
        {
            URBNode.start = start + count[0] + count[1] + count[2] + count[3] + count[4] + count[5];
            URBNode.end = start + count[0] + count[1] + count[2] + count[3] + count[4] + count[5] + count[6] - 1;
        }
        
        if (count[7] > 0)
        {
            URFNode.start = start + count[0] + count[1] + count[2] + count[3] + count[4] + count[5] + count[6];
            URFNode.end = end;
        }
        ConstructQuadTreeKernel<<<8, BLOCK_SIZE>>>(node, buffer, bodies, nodeIndex * 8 + 1, nNodes, nBodies, leafLimit);
    }
}




/*
----------------------------------------------------------------------------------------
COMPUTE FORCE
----------------------------------------------------------------------------------------
*/
__device__ float getDistance(Vector pos1, Vector pos2)
{

    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2) + pow(pos1.z - pos2.z, 2));
}

__device__ bool isCollide(Body &b1, Vector cm)
{
    return b1.radius * 2 + COLLISION_TH > getDistance(b1.position, cm);
}

__device__ void ComputeForce(Node *node, Body *bodies, int nodeIndex, int bodyIndex, int nNodes, int nBodies, int leafLimit, float width)
{

    if (nodeIndex >= nNodes)
    {
        return;
    }
    Node curNode = node[nodeIndex];
    Body bi = bodies[bodyIndex];
    if (curNode.isLeaf)
    {
        if (curNode.centerMass.x != -1 && !isCollide(bi, curNode.centerMass))
        {
            Vector rij = {curNode.centerMass.x - bi.position.x, curNode.centerMass.y - bi.position.y, curNode.centerMass.z - bi.position.z};
            float r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z) + (E * E));
            float f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f, rij.z * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
            bodies[bodyIndex].acceleration.z += (force.z / bi.mass);
        }
        return;
    }

    float sd = width / getDistance(bi.position, curNode.centerMass);
    if (sd < THETA)
    {
        if (!isCollide(bi, curNode.centerMass))
        {
            Vector rij = {curNode.centerMass.x - bi.position.x, curNode.centerMass.y - bi.position.y, curNode.centerMass.z - bi.position.z};
            float r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z) + (E * E));
            float f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f, rij.z * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
            bodies[bodyIndex].acceleration.z += (force.z / bi.mass);
        }

        return;
    }

    ComputeForce(node, bodies, (nodeIndex * 8) + 1, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 8) + 2, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 8) + 3, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 8) + 4, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 8) + 5, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 8) + 6, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 8) + 7, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 8) + 8, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
}



__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float width = node[0].maxCorner.x - node[0].minCorner.x;
    
    if (i < nBodies)
    {
        Body &bi = bodies[i];
        if (bi.isDynamic)
        {
            bi.acceleration = {0.0, 0.0, 0.0};
            ComputeForce(node, bodies, 0, i, nNodes, nBodies, leafLimit, width);
        }
    }
}





void saveToFile(const char* filename, Body* h_b, Node* h_node, int nBodies, int nNodes) {
    ofstream file(filename, ios::out | ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file for writing!" << endl;
        return;
    }

    // Write nBodies and nNodes
    file.write(reinterpret_cast<char*>(&nBodies), sizeof(nBodies));
    file.write(reinterpret_cast<char*>(&nNodes), sizeof(nNodes));

    // Write positions of bodies
    for (int i = 0; i < nBodies; ++i) {
        file.write(reinterpret_cast<char*>(&h_b[i].position.x), sizeof(h_b[i].position.x));
        file.write(reinterpret_cast<char*>(&h_b[i].position.y), sizeof(h_b[i].position.y));
        file.write(reinterpret_cast<char*>(&h_b[i].position.z), sizeof(h_b[i].position.z));
    }

    // Write boundaries of nodes
    for (int i = 0; i < nNodes; ++i) {
        file.write(reinterpret_cast<char*>(&h_node[i].minCorner.x), sizeof(h_node[i].minCorner.x));
        file.write(reinterpret_cast<char*>(&h_node[i].minCorner.y), sizeof(h_node[i].minCorner.y));
        file.write(reinterpret_cast<char*>(&h_node[i].minCorner.z), sizeof(h_node[i].minCorner.z));
        file.write(reinterpret_cast<char*>(&h_node[i].maxCorner.x), sizeof(h_node[i].maxCorner.x));
        file.write(reinterpret_cast<char*>(&h_node[i].maxCorner.y), sizeof(h_node[i].maxCorner.y));
        file.write(reinterpret_cast<char*>(&h_node[i].maxCorner.z), sizeof(h_node[i].maxCorner.z));
        file.write(reinterpret_cast<char*>(&h_node[i].start), sizeof(h_node[i].start));
        file.write(reinterpret_cast<char*>(&h_node[i].end), sizeof(h_node[i].end));
    }

    file.close();
}




void saveBodyToFile(const char* filename, Body* h_b, int nBodies)
{
    ofstream file(filename, ios::out | ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file for writing!" << endl;
        return;
    }

    // Write nBodies
    file.write(reinterpret_cast<char*>(&nBodies), sizeof(nBodies));

    // Write bodies
    for (int i = 0; i < nBodies; ++i)
    {
        file.write(reinterpret_cast<char*>(&h_b[i]), sizeof(Body));
    }

    file.close();
}




int main()
{

  int n = 4000000; // number of particles.
  int nBodies = n;

  int MAX_NODES = 1000000;
  int N_LEAF = 500000;

  int nNodes = MAX_NODES;
  int leafLimit = MAX_NODES - N_LEAF;
  
  Body *h_b, *d_b, *d_b_buffer;
  Node *h_node, *d_node;
  
  int *d_mutex;
  
  h_b = new Body[n];
  h_node = new Node[nNodes];

  cudaMalloc((void **)&d_b, sizeof(Body) * n);
  cudaMalloc((void **)&d_node, sizeof(Node) * nNodes);
  cudaMalloc((void **)&d_mutex, sizeof(int) * nNodes);
  cudaMalloc((void **)&d_b_buffer, sizeof(Body) * n); // n = nBodies


  //--- preparing bodies (bodies are actually particles!) ---
  //random_device rd;  // Obtain a random number from hardware
  mt19937 eng(42); // Seed the generator
  uniform_real_distribution<> distr(-1.0, 1.0); // Define the range
  
  for (int i = 0; i < n; i++)
  {
    h_b[i].position.x = distr(eng);
    h_b[i].position.y = distr(eng);
    h_b[i].position.z = distr(eng);
    
    h_b[i].mass = 1.0;
  }
  
  //--- copying particles from host to device
  cudaMemcpy(d_b, h_b, nBodies * sizeof(Body), cudaMemcpyHostToDevice);
  
  cout << "h_node.minCorner.x = " << h_node[0].minCorner.x << endl;
  cout << "h_node.minCorner.y = " << h_node[0].minCorner.y << endl;
  cout << "h_node.minCorner.z = " << h_node[0].minCorner.z << endl;
  cout << "h_node.maxCorner.x = " << h_node[0].maxCorner.x << endl;
  cout << "h_node.maxCorner.y = " << h_node[0].maxCorner.y << endl;
  cout << "h_node.maxCorner.z = " << h_node[0].maxCorner.z << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;
  
  int blockSize = BLOCK_SIZE;
  dim3 gridSize = ceil((float)nNodes / blockSize);
  ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);
  cudaDeviceSynchronize();

  cudaMemcpy(h_node, d_node, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
  cout << "h_node.minCorner.x = " << h_node[0].minCorner.x << endl;
  cout << "h_node.minCorner.y = " << h_node[0].minCorner.y << endl;
  cout << "h_node.minCorner.z = " << h_node[0].minCorner.z << endl;
  cout << "h_node.maxCorner.x = " << h_node[0].maxCorner.x << endl;
  cout << "h_node.maxCorner.y = " << h_node[0].maxCorner.y << endl;
  cout << "h_node.maxCorner.z = " << h_node[0].maxCorner.z << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;


  blockSize = BLOCK_SIZE;
  gridSize = ceil((float)nBodies / blockSize);
  ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);
  cudaDeviceSynchronize();

  cudaMemcpy(h_node, d_node, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
  cout << "h_node.minCorner.x = " << h_node[0].minCorner.x << endl;
  cout << "h_node.minCorner.y = " << h_node[0].minCorner.y << endl;
  cout << "h_node.minCorner.z = " << h_node[0].minCorner.z << endl;
  cout << "h_node.maxCorner.x = " << h_node[0].maxCorner.x << endl;
  cout << "h_node.maxCorner.y = " << h_node[0].maxCorner.y << endl;
  cout << "h_node.maxCorner.z = " << h_node[0].maxCorner.z << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;
  
  blockSize = BLOCK_SIZE;
  gridSize = ceil((float)nBodies / blockSize);
  ConstructQuadTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
  cudaDeviceSynchronize();

  //----- example prints --
  cudaMemcpy(h_node, d_node, sizeof(Node) * nNodes, cudaMemcpyDeviceToHost);
  Node node_1 = h_node[100];
  cout << "start = " << node_1.start << endl;
  cout << "end = " << node_1.end << endl;


  auto T_hh = std::chrono::high_resolution_clock::now();

  blockSize = 32; // perhaps because of the warp use in ComputeForce function!
  gridSize = ceil((float)nBodies / blockSize);
  ComputeForceKernel<<<gridSize, blockSize>>>(d_node, d_b, nNodes, nBodies, leafLimit);
  cudaDeviceSynchronize();
  
  auto end_hh = std::chrono::high_resolution_clock::now();
  auto elapsed_hh = std::chrono::duration_cast<std::chrono::nanoseconds>(end_hh - T_hh);
  cout << "T_h = " << elapsed_hh.count() * 1e-9 << endl;
  
  cudaMemcpy(h_b, d_b, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost);
  
  Body b1 = h_b[100];
  
  float accx = b1.acceleration.x;
  float accy = b1.acceleration.y;
  float accz = b1.acceleration.z;
  
  printf("(accx, accy, accz) = %f, %f, %f\n", accx, accy, accz);
  cout << "b1.position.x = " << b1.position.x << endl;
  cout << "b1.position.y = " << b1.position.y << endl;
  cout << "b1.position.z = " << b1.position.z << endl;


  // Save h_b to a binary file
  for (int i = 0; i < n; i++)
  {
    h_b[i].acceleration.x = 0.0;
    h_b[i].acceleration.y = 0.0;
    h_b[i].acceleration.z = 0.0;
  }

  saveBodyToFile("h_b.bin", h_b, nBodies);


  //----- Output to a file -----
  saveToFile("BH.bin", h_b, h_node, nBodies, nNodes);
  


}



