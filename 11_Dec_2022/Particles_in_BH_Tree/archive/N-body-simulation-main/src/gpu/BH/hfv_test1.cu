%%writefile test.cu

#include <iostream>
#include <cmath>
#include <random>
#include <fstream>

using namespace std;

const int BLOCK_SIZE = 256;

//------ Vector -------
struct Vector
{
  double x;
  double y;
};


//------ Body -------
struct Body
{
  bool isDynamic;
  double mass;
  double radius;
  Vector position;
  Vector velocity;
  Vector acceleration;
};


//------ Node -------
struct Node
{
  Vector topLeft;
  Vector botRight;
  Vector centerMass;
  double totalMass;
  bool isLeaf;
  int start;
  int end;
};


//------ BarnesHutCuda -------
class BarnesHutCuda
{
    int nBodies;
    int nNodes;
    int leafLimit;

    Body *h_b;
    Node *h_node;

    Body *d_b;
    Body *d_b_buffer;
    Node *d_node;
    int *d_mutex;

    void initRandomBodies();
    void initSpiralBodies();
    void initCollideGalaxy();
    void initSolarSystem();
    void setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration);
    void resetCUDA();
    void computeBoundingBoxCUDA();
    void constructQuadTreeCUDA();
    void computeForceCUDA();

public:
    BarnesHutCuda(int n);
    ~BarnesHutCuda();
    void update();
    void setup(int sim);
    void readDeviceBodies();
    Body *getBodies();
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
        node[b].topLeft = {INFINITY, -INFINITY};
        node[b].botRight = {-INFINITY, INFINITY};
        node[b].centerMass = {-1, -1};
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

    __shared__ double topLeftX[BLOCK_SIZE];
    __shared__ double topLeftY[BLOCK_SIZE];
    __shared__ double botRightX[BLOCK_SIZE];
    __shared__ double botRightY[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = tx + blockIdx.x * blockDim.x;

    topLeftX[tx] = INFINITY;
    topLeftY[tx] = -INFINITY;
    botRightX[tx] = -INFINITY;
    botRightY[tx] = INFINITY;

    __syncthreads();

    if (b < nBodies)
    {
        Body body = bodies[b];
        topLeftX[tx] = body.position.x;
        topLeftY[tx] = body.position.y;
        botRightX[tx] = body.position.x;
        botRightY[tx] = body.position.y;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            topLeftX[tx] = fminf(topLeftX[tx], topLeftX[tx + s]);
            topLeftY[tx] = fmaxf(topLeftY[tx], topLeftY[tx + s]);
            botRightX[tx] = fmaxf(botRightX[tx], botRightX[tx + s]);
            botRightY[tx] = fminf(botRightY[tx], botRightY[tx + s]);
        }
    }

    if (tx == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0)
            ;
        node[0].topLeft.x = fminf(node[0].topLeft.x, topLeftX[0] - 0.01);
        node[0].topLeft.y = fmaxf(node[0].topLeft.y, topLeftY[0] + 0.01);
        node[0].botRight.x = fmaxf(node[0].botRight.x, botRightX[0] + 0.01);
        node[0].botRight.y = fminf(node[0].botRight.y, botRightY[0] - 0.01);
        atomicExch(mutex, 0);
    }
}



/*
----------------------------------------------------------------------------------------
CONSTRUCT QUAD TREE
----------------------------------------------------------------------------------------
*/
__device__ int getQuadrant(Vector topLeft, Vector botRight, double x, double y)
{

    if ((topLeft.x + botRight.x) / 2 >= x)
    {
        // Indicates topLeftTree
        if ((topLeft.y + botRight.y) / 2 <= y)
        {
            return 2;
        }
        // Indicates botLeftTree
        else
        {
            return 3;
        }
    }
    else
    {
        // Indicates topRightTree
        if ((topLeft.y + botRight.y) / 2 <= y)
        {
            return 1;
        }
        // Indicates botRightTree
        else
        {
            return 4;
        }
    }
}

__device__ void UpdateChildBound(Vector &tl, Vector &br, Node &childNode, int quadrant)
{

    if (quadrant == 1)
    {
        childNode.topLeft = {(tl.x + br.x) / 2, tl.y};
        childNode.botRight = {br.x, (tl.y + br.y) / 2};
    }
    else if (quadrant == 2)
    {
        childNode.topLeft = {tl.x, tl.y};
        childNode.botRight = {(tl.x + br.x) / 2, (tl.y + br.y) / 2};
    }
    else if (quadrant == 3)
    {
        childNode.topLeft = {tl.x, (tl.y + br.y) / 2};
        childNode.botRight = {(tl.x + br.x) / 2, br.y};
    }
    else
    {
        childNode.topLeft = {(tl.x + br.x) / 2, (tl.y + br.y) / 2};
        childNode.botRight = {br.x, br.y};
    }
}

__device__ void warpReduce(volatile double *totalMass, volatile double2 *centerMass, int tx)
{
    totalMass[tx] += totalMass[tx + 32];
    centerMass[tx].x += centerMass[tx + 32].x;
    centerMass[tx].y += centerMass[tx + 32].y;
    
    totalMass[tx] += totalMass[tx + 16];
    centerMass[tx].x += centerMass[tx + 16].x;
    centerMass[tx].y += centerMass[tx + 16].y;
    
    totalMass[tx] += totalMass[tx + 8];
    centerMass[tx].x += centerMass[tx + 8].x;
    centerMass[tx].y += centerMass[tx + 8].y;
    
    totalMass[tx] += totalMass[tx + 4];
    centerMass[tx].x += centerMass[tx + 4].x;
    centerMass[tx].y += centerMass[tx + 4].y;
    
    totalMass[tx] += totalMass[tx + 2];
    centerMass[tx].x += centerMass[tx + 2].x;
    centerMass[tx].y += centerMass[tx + 2].y;
    
    totalMass[tx] += totalMass[tx + 1];
    centerMass[tx].x += centerMass[tx + 1].x;
    centerMass[tx].y += centerMass[tx + 1].y;
}

__device__ void ComputeCenterMass(Node &curNode, Body *bodies, double *totalMass, double2 *centerMass, int start, int end)
{
    int tx = threadIdx.x;
    int total = end - start + 1;
    int sz = ceil((double)total / blockDim.x);
    int s = tx * sz + start;
    double M = 0.0;
    double2 R = make_double2(0.0, 0.0);

    for (int i = s; i < s + sz; ++i)
    {
        if (i <= end)
        {
            Body &body = bodies[i];
            M += body.mass;
            R.x += body.mass * body.position.x;
            R.y += body.mass * body.position.y;
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
        curNode.totalMass = totalMass[0];
        curNode.centerMass = {centerMass[0].x, centerMass[0].y};
    }
}

__device__ void CountBodies(Body *bodies, Vector topLeft, Vector botRight, int *count, int start, int end, int nBodies)
{
    int tx = threadIdx.x;
    if (tx < 4)
        count[tx] = 0;
    __syncthreads();

    for (int i = start + tx; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int q = getQuadrant(topLeft, botRight, body.position.x, body.position.y);
        atomicAdd(&count[q - 1], 1);
    }

    __syncthreads();
}

__device__ void ComputeOffset(int *count, int start)
{
    int tx = threadIdx.x;
    if (tx < 4)
    {
        int offset = start;
        for (int i = 0; i < tx; ++i)
        {
            offset += count[i];
        }
        count[tx + 4] = offset;
    }
    __syncthreads();
}

__device__ void GroupBodies(Body *bodies, Body *buffer, Vector topLeft, Vector botRight, int *count, int start, int end, int nBodies)
{
    int *count2 = &count[4];
    for (int i = start + threadIdx.x; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int q = getQuadrant(topLeft, botRight, body.position.x, body.position.y);
        int dest = atomicAdd(&count2[q - 1], 1);
        buffer[dest] = body;
    }
    __syncthreads();
}

__global__ void ConstructQuadTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit)
{
    __shared__ int count[8];
    __shared__ double totalMass[BLOCK_SIZE];
    __shared__ double2 centerMass[BLOCK_SIZE];
    int tx = threadIdx.x;
    nodeIndex += blockIdx.x;

    if (nodeIndex > 29000)
      printf("nodeIndex, blockIdx.x = %d, %d\n", nodeIndex, blockIdx.x);

    if (nodeIndex >= nNodes)
        return;

    Node &curNode = node[nodeIndex];
    int start = curNode.start;
    int end = curNode.end;
    
    Vector topLeft = curNode.topLeft;
    Vector botRight = curNode.botRight;

    if (start == -1 && end == -1)
        return;
    
    if (end - start < 20) //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

    CountBodies(bodies, topLeft, botRight, count, start, end, nBodies);
    ComputeOffset(count, start);
    GroupBodies(bodies, buffer, topLeft, botRight, count, start, end, nBodies);

    if (tx == 0)
    {
        Node &topLNode = node[(nodeIndex * 4) + 2];
        Node &topRNode = node[(nodeIndex * 4) + 1];
        Node &botLNode = node[(nodeIndex * 4) + 3];
        Node &botRNode = node[(nodeIndex * 4) + 4];

        UpdateChildBound(topLeft, botRight, topLNode, 2);
        UpdateChildBound(topLeft, botRight, topRNode, 1);
        UpdateChildBound(topLeft, botRight, botLNode, 3);
        UpdateChildBound(topLeft, botRight, botRNode, 4);

        curNode.isLeaf = false;

        if (count[0] > 0)
        {
            topRNode.start = start;
            topRNode.end = start + count[0] - 1;
        }

        if (count[1] > 0)
        {
            topLNode.start = start + count[0];
            topLNode.end = start + count[0] + count[1] - 1;
        }

        if (count[2] > 0)
        {
            botLNode.start = start + count[0] + count[1];
            botLNode.end = start + count[0] + count[1] + count[2] - 1;
        }

        if (count[3] > 0)
        {
            botRNode.start = start + count[0] + count[1] + count[2];
            botRNode.end = end;
        }
        ConstructQuadTreeKernel<<<4, BLOCK_SIZE>>>(node, buffer, bodies, nodeIndex * 4 + 1, nNodes, nBodies, leafLimit);
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
    }

    // Write boundaries of nodes
    for (int i = 0; i < nNodes; ++i) {
        file.write(reinterpret_cast<char*>(&h_node[i].topLeft.x), sizeof(h_node[i].topLeft.x));
        file.write(reinterpret_cast<char*>(&h_node[i].topLeft.y), sizeof(h_node[i].topLeft.y));
        file.write(reinterpret_cast<char*>(&h_node[i].botRight.x), sizeof(h_node[i].botRight.x));
        file.write(reinterpret_cast<char*>(&h_node[i].botRight.y), sizeof(h_node[i].botRight.y));
    }

    file.close();
}





int main()
{

  int n = 100000; // number of particles.
  int nBodies = n;

  int MAX_NODES = 30000;
  int N_LEAF = 15000;

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
    
    h_b[i].mass = 1.0;
  }
  
  //--- copying particles from host to device
  cudaMemcpy(d_b, h_b, nBodies * sizeof(Body), cudaMemcpyHostToDevice);
  
  cout << "h_node.topLeft.x = " << h_node[0].topLeft.x << endl;
  cout << "h_node.topLeft.y = " << h_node[0].topLeft.y << endl;
  cout << "h_node.botRight.x = " << h_node[0].botRight.x << endl;
  cout << "h_node.botRight.y = " << h_node[0].botRight.y << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;
  
  int blockSize = BLOCK_SIZE;
  dim3 gridSize = ceil((float)nNodes / blockSize);
  ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);

  cudaMemcpy(h_node, d_node, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
  cout << "h_node.topLeft.x = " << h_node[0].topLeft.x << endl;
  cout << "h_node.topLeft.y = " << h_node[0].topLeft.y << endl;
  cout << "h_node.botRight.x = " << h_node[0].botRight.x << endl;
  cout << "h_node.botRight.y = " << h_node[0].botRight.y << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;


  blockSize = BLOCK_SIZE;
  gridSize = ceil((float)nBodies / blockSize);
  ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);

  cudaMemcpy(h_node, d_node, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
  cout << "h_node.topLeft.x = " << h_node[0].topLeft.x << endl;
  cout << "h_node.topLeft.y = " << h_node[0].topLeft.y << endl;
  cout << "h_node.botRight.x = " << h_node[0].botRight.x << endl;
  cout << "h_node.botRight.y = " << h_node[0].botRight.y << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;
  
  blockSize = BLOCK_SIZE;
  gridSize = ceil((float)nBodies / blockSize);
  ConstructQuadTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);

  //----- example prints --
  cudaMemcpy(h_node, d_node, sizeof(Node) * nNodes, cudaMemcpyDeviceToHost);
  Node node_1 = h_node[50];
  cout << "start = " << node_1.start << endl;
  cout << "end = " << node_1.end << endl;




  //----- Output to a file -----
  saveToFile("BH.bin", h_b, h_node, nBodies, nNodes);
  


}


