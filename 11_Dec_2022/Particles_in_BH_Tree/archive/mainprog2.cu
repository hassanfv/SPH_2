%%writefile test.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>

using namespace std;

#define BLOCK_DIM 256
typedef float FLOAT;


const int N = 100000;


struct FLOAT3
{
  float x, y, z;
};



//===== Octree_node
struct Octree_node
{
  // Attributes of an Octree node
  FLOAT3 center;       // Center of the node in 3D space
  float width;         // Width of the node (assuming cubic volume)
  int points_begin_idx; // Range of points that belong to this node (begin index)
  int points_end_idx;  // Range of points that belong to this node (end index)
  int children_idx;    // Index to the first child node if this node is subdivided
  FLOAT3 center_mass;  // Center of mass for the node based on the contained points

  // Constructor
  __host__ __device__ Octree_node() : center{0, 0, 0}, width(0), points_begin_idx(-1), points_end_idx(-1), children_idx(-1), center_mass{0, 0, 0} {}

  // Set the center of the node
  __host__ __device__ void set_center(float x, float y, float z)
  {
    center.x = x;
    center.y = y;
    center.z = z;
  }

  // Set the width of the node
  __host__ __device__ void set_width(float w)
  {
    width = w;
  }

  // Set the range of points indices that belong to this node
  __host__ __device__ void set_range(int begin, int end)
  {
    points_begin_idx = begin;
    points_end_idx = end;
  }

  // Set the index of the first child node
  __host__ __device__ void set_children(int idx)
  {
    children_idx = idx;
  }

  // Set the center of mass of the node
  __host__ __device__ void set_center_mass(float x, float y, float z)
  {
    center_mass.x = x;
    center_mass.y = y;
    center_mass.z = z;
  }

  // Get the center of the node
  __host__ __device__ FLOAT3 get_center() const
  {
    return center;
  }

  __host__ __device__ float get_width() const
  {
    return width;
  }

  __host__ __device__ int get_points_begin() const
  {
    return points_begin_idx;
  }

  __host__ __device__ int get_points_end() const
  {
    return points_end_idx;
  }

  // Get the index of the first child node
  __host__ __device__ int children() const
  {
    return children_idx;
  }

  // Get the number of points in this node
  __host__ __device__ int num_points() const
  {
    return points_end_idx - points_begin_idx;
  }

  // Get the center of mass of the node
  __host__ __device__ FLOAT3 get_center_mass() const
  {
    return center_mass;
  }
};




// Define a helper function to generate random floats in [-1, 1]
float random_float()
{
    return 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
}



// Assuming BLOCK_DIM is defined and is the number of threads per block.
__global__ void maxCoord(const FLOAT3* dev_unc_pos, FLOAT* dev_result, int num_points)
{
    extern __shared__ FLOAT shared_max[]; // Shared memory for inter-thread communication within a block
    int tid = threadIdx.x; // Thread ID within the block
    int i = blockIdx.x * blockDim.x + tid; // Global index for the entire grid
    
    FLOAT max_val = -FLT_MAX; // Initialize with the smallest float value

    // Iterate over all elements assigned to this thread in the grid, striding by the total number of threads
    while(i < num_points)
    {
        FLOAT3 pos = dev_unc_pos[i]; // Get the position at index i
        // Determine the maximum value across all dimensions
        FLOAT max_coord = fmaxf(fmaxf(pos.x, pos.y), pos.z);
        max_val = fmaxf(max_val, max_coord); // Update the current max_val
        i += blockDim.x * gridDim.x; // Move to the next element
    }

    // Store the found max value in shared memory
    shared_max[tid] = max_val;

    __syncthreads(); // Synchronize threads within the block

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads(); // Ensure all accesses to shared memory have completed
    }

    // The first thread in the block writes the result to the global memory
    if (tid == 0)
    {
        atomicMax((int*)dev_result, __float_as_int(shared_max[0]));
    }
}


//===== init_nodes
__global__ void init_nodes(int *points, int num_points)
{

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  for(int iter = threadIdx.x; iter<num_points; iter+=blockDim.x)
  {
    points[iter] = i;
  }
}



//===== Parameters
struct Parameters
{
  int depth;          // Current depth in the octree
  int max_depth;      // Maximum allowable depth for the octree
  int min_points;     // Minimum number of points required to split a node further
  int point_selector; // Selector to determine which points buffer to use

  // Constructor
  __host__ __device__ Parameters(int nbead)
  {
    // Initialize the parameters with appropriate values
    depth = 0; // Starting depth could be 0
    max_depth = 10; // You may change it!!!
    min_points = 1; // You may change it!!!
    point_selector = 0; // Initial value for point selector
  }

  // Overloaded constructor for creating a new parameters object for deeper levels
  __host__ __device__ Parameters(const Parameters& other, bool incrementDepth)
  {
    // Copy all parameters from 'other'
    depth = other.depth;
    max_depth = other.max_depth;
    min_points = other.min_points;
    point_selector = other.point_selector;

    // Increment depth if required
    if (incrementDepth)
    {
      depth++;
    }
  }
};





int main()
{

  int nbead = N;

  dim3 threads(BLOCK_DIM, 1, 1);

  dim3 grid((int) ceil((nbead + 1.0)/(FLOAT) threads.x), 1, 1);

  int GRID_DIM = grid.x;

  FLOAT *dev_result;
  
  FLOAT result[GRID_DIM];
  
  FLOAT init_value = -100;
  for (int i = 0; i < GRID_DIM; i++)
  {
    result[i] = init_value;
  }

  cudaMalloc((void **)&dev_result, GRID_DIM*sizeof(FLOAT));
  
  cudaMemcpy(dev_result, result, GRID_DIM * sizeof(FLOAT), cudaMemcpyHostToDevice);


  cout << "threads.x = " << threads.x << endl;
  cout << "grid.x = " << grid.x << endl;

  srand(time(0)); // Seed for random number generation

  // Allocate host array
  FLOAT3* unc_pos = new FLOAT3[N];

  // Initialize the array with random numbers
  for (int i = 0; i < N; ++i)
  {
    unc_pos[i].x = random_float();
    unc_pos[i].y = random_float();
    unc_pos[i].z = random_float();
  }

  // Allocate memory on the device
  FLOAT3* dev_unc_pos;
  cudaMalloc((void**)&dev_unc_pos, N * sizeof(FLOAT3));

  // Copy data from host to device
  cudaMemcpy(dev_unc_pos, unc_pos, N * sizeof(FLOAT3), cudaMemcpyHostToDevice);


  maxCoord<<<grid, threads, GRID_DIM*sizeof(FLOAT)>>>(dev_unc_pos, dev_result, nbead);
  cudaDeviceSynchronize();
  
  cudaMemcpy(result, dev_result, GRID_DIM*sizeof(FLOAT), cudaMemcpyDeviceToHost);

  cout << "result[0] = " << result[0] << endl;


  Octree_node root;

  root.set_range(0, nbead);
  root.set_width(2.0f * ceil(result[0]));

  printf("Ranges: (%d, %d)\n", root.get_points_begin(), root.get_points_end());
  printf("width: %f\n", root.get_width());
  
  int expected_number_of_nodes = 1000;
  
  Octree_node *all_nodes;
  cudaMalloc((void**)&all_nodes, sizeof(Octree_node) * expected_number_of_nodes);
  cudaMemcpy(all_nodes, &root, sizeof(Octree_node), cudaMemcpyHostToDevice);

  int* dev_buffer1;
  int* dev_buffer2;

  // Allocate memory on the device for dev_buffer1
  cudaError_t err1 = cudaMalloc((void**)&dev_buffer1, N * sizeof(int));
  if (err1 != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err1));
  }

  // Allocate memory on the device for dev_buffer2
  cudaError_t err2 = cudaMalloc((void**)&dev_buffer2, N * sizeof(int));
  if (err2 != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err2));
      cudaFree(dev_buffer1);
  }


  init_nodes<<<grid, threads>>>(dev_buffer1, nbead);
  
  Parameters params(nbead);
  
  int num_nodes_host[1] = {1};

  int* num_nodes;
  cudaMalloc((void**)&num_nodes, sizeof(int));
  cudaMemcpy(num_nodes, num_nodes_host, sizeof(int), cudaMemcpyHostToDevice);



  
  
  
  


  cudaFree(dev_unc_pos);
  cudaFree(dev_buffer1);
  cudaFree(dev_buffer2);
  delete[] unc_pos;

}

