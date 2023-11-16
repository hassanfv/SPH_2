%%writefile test.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <fstream>

using namespace std;

#define BLOCK_DIM 2
//typedef float FLOAT;


int N = 5;


struct FLOAT2
{
  float x, y;
};


//===== Octree_node
struct Octree_node
{
  // Attributes of an Octree node
  FLOAT2 center;       // Center of the node in 3D space
  float width;         // Width of the node (assuming cubic volume)
  int points_begin_idx; // Range of points that belong to this node (begin index)
  int points_end_idx;  // Range of points that belong to this node (end index)
  int children_idx;    // Index to the first child node if this node is subdivided
  FLOAT2 center_mass;  // Center of mass for the node based on the contained points

  // Constructor
  __host__ __device__ Octree_node() : center{0, 0}, width(0), points_begin_idx(-1), points_end_idx(-1), children_idx(-1), center_mass{0, 0} {}

  // Set the center of the node
  __host__ __device__ void set_center(float x, float y)
  {
    center.x = x;
    center.y = y;
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
  __host__ __device__ void set_center_mass(float x, float y)
  {
    center_mass.x = x;
    center_mass.y = y;
  }

  // Get the center of the node
  __host__ __device__ FLOAT2 get_center() const
  {
    return center;
  }

  __host__ __device__ float get_width() const
  {
    return width;
  }

  __host__ __device__ int points_begin() const
  {
    return points_begin_idx;
  }

  __host__ __device__ int points_end() const
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
  __host__ __device__ FLOAT2 get_center_mass() const
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
__global__ void maxCoord(const FLOAT2* dev_unc_pos, float* dev_result, int num_points)
{
    extern __shared__ float shared_max[]; // Shared memory for inter-thread communication within a block
    int tid = threadIdx.x; // Thread ID within the block
    int i = blockIdx.x * blockDim.x + tid; // Global index for the entire grid
    
    float max_val = -100.0; // Initialize with the smallest float value

    // Iterate over all elements assigned to this thread in the grid, striding by the total number of threads
    while(i < num_points)
    {
        FLOAT2 pos = dev_unc_pos[i]; // Get the position at index i
        // Determine the maximum value across all dimensions
        float max_coord = fmaxf(pos.x, pos.y);
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
    max_depth = 1000; // You may change it!!!
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
      point_selector = 1;
    }
  }
};





//===== check_points
__device__ bool check_points(Octree_node &node, int *points1, int *points2, int num_points, Parameters params)
{

  if(params.depth >= params.max_depth || num_points <= params.min_points)
  {
    if(params.point_selector == 1)
    {
      int it = node.points_begin();
      int end = node.points_end();
      
      for(it += threadIdx.x; it < end; it += blockDim.x)
      {
        points1[it] = points2[it];
      }
    }
    
    return true;
  }
  
  return false;
}


//===== count_points
__device__ void count_points(const int *in_points, FLOAT2* dev_unc_pos, int *smem, int range_begin, int range_end, FLOAT2 center)
{

  if(threadIdx.x < 4) smem[threadIdx.x] = 0;

  __syncthreads();

  for(int iter=range_begin+threadIdx.x; iter<range_end; iter+=blockDim.x)
  {
    FLOAT2 p = dev_unc_pos[in_points[iter]];
    int x = p.x < center.x ? 0 : 1;
    int y = p.y < center.y ? 0 : 1;
    int i = x*2 + y;
    atomicAdd(&smem[i], 1);
  }
  __syncthreads();
}


//===== scan_offsets
__device__ void scan_offsets(int node_points_begin, int* smem)
{

  int *smem2 = &smem[4];
  
  if(threadIdx.x == 0)
  {
    for(int i = 0; i < 4; i++)
    {
      smem2[i] = i == 0 ? 0 : smem2[i-1] + smem[i-1];
    }
    for (int i = 0; i < 4; i++)
    {
      smem2[i] += node_points_begin;
    }
  }
  __syncthreads();
}


//===== reorder_points
__device__ void reorder_points(int *out_points, const int *in_points, FLOAT2 *dev_unc_pos, int *smem, int range_begin, int range_end, FLOAT2 center)
{

  int *smem2 = &smem[4];
  
  for(int iter = range_begin+threadIdx.x; iter<range_end; iter+=blockDim.x)
  {
    FLOAT2 p = dev_unc_pos[in_points[iter]];

    int x = p.x < center.x ? 0 : 1;
    int y = p.y < center.y ? 0 : 1;
    
    int i = x*2 + y;
    
    int dest = atomicAdd(&smem2[i], 1);
    
    out_points[dest] = in_points[iter];
  }
  __syncthreads();
}



//===== prepare_children
__device__ void prepare_children(Octree_node *children, Octree_node &node, int *smem)
{

  const FLOAT2 center = node.get_center();

  float half = node.get_width() / 2.0f;
  float quarter = half / 2.0f;

  smem[3] = node.points_begin();

  for(int i = 0; i < 4; i++)
  {
    int xf, yf;

    yf = i % 2;
    if (yf == 0) yf = -1;

    xf = i / 2;
    if (xf == 0) xf = -1;

    children[i].set_center(center.x + quarter * xf, center.y + quarter * yf);
    children[i].set_width(half);
    children[i].set_range(smem[3+i], smem[4+i]);
  }
}



//===== build_octree_kernel
__global__ void build_octree_kernel(Octree_node *nodes, Octree_node *all_nodes, int *num_nodes, int *points1, int *points2, FLOAT2 *dev_unc_pos, Parameters params)
{

  __shared__ int smem[8];
  
  if(threadIdx.x == blockDim.x-1)
  {
    Octree_node &node = nodes[blockIdx.x];
    
    
    int num_points = node.num_points();
    printf("(blockDim.x,  num_points) = (%d, %d)\n", blockDim.x, num_points);
    bool exit = check_points(node, points1, points2, num_points, params);
    
    if(exit) return;
    
    FLOAT2 center = node.get_center();
    
    int range_begin = node.points_begin();
    int range_end = node.points_end();
    
    int *in_points = params.point_selector == 0 ? points1 : points2;
    
    int *out_points = params.point_selector == 0 ? points2 : points1;
    
    count_points(in_points, dev_unc_pos, smem, range_begin, range_end, center);
    
    scan_offsets(node.points_begin(), smem);
    
    reorder_points(out_points, in_points, dev_unc_pos, smem, range_begin, range_end, center);

    int next_node = atomicAdd(num_nodes, 4);
    
    printf("next_node = %d\n", next_node);
    printf("point_selector = %d\n", params.point_selector);
    printf("\n");
    
    node.set_children(next_node);
    
    Octree_node *children = &all_nodes[next_node];
    
    prepare_children(children, node, smem);
    
    build_octree_kernel<<<4, blockDim.x, 8*sizeof(int)>>>(children, all_nodes, num_nodes, in_points, out_points, dev_unc_pos, Parameters(params, true));
  }
}


//===== calculate_centers_mass
__global__ void calculate_centers_mass(Octree_node *all_nodes, int *num_nodes, int *points, FLOAT2 *dev_unc_pos)
{

  float x = 0.0, y = 0.0;
  
  Octree_node node;
  
  for(int iter = threadIdx.x; iter<num_nodes[0]; iter+=blockDim.x)
  {
    node = all_nodes[iter];
    int begin = node.points_begin();
    int end = node.points_end();
    
    for (int i = begin; i < end; i++)
    {
      x += dev_unc_pos[points[i]].x;
      y += dev_unc_pos[points[i]].y;
    }
    
    x /= node.num_points();
    y /= node.num_points();
    
    node.set_center_mass(x/node.num_points(), y/node.num_points());
  }
}







int main()
{

  int nbead = N;

  dim3 threads(BLOCK_DIM, 1, 1);

  dim3 grid((int) ceil((nbead + 1.0)/(float) threads.x), 1, 1);

  int GRID_DIM = grid.x;

  float *dev_result;
  
  float result[GRID_DIM];
  
  float init_value = -100;
  for (int i = 0; i < GRID_DIM; i++)
  {
    result[i] = init_value;
  }

  cudaMalloc((void **)&dev_result, GRID_DIM*sizeof(float));
  
  cudaMemcpy(dev_result, result, GRID_DIM * sizeof(float), cudaMemcpyHostToDevice);


  cout << "threads.x = " << threads.x << endl;
  cout << "grid.x = " << grid.x << endl;

  srand(42); // Seed for random number generation

  // Allocate host array
  FLOAT2* unc_pos = new FLOAT2[N];

  float xtmp[] = {0.1, 0.2, 0.9, -0.2, -0.4};
  float ytmp[] = {0.1, 0.2, 0.9, -0.2, -0.4};

  // Initialize the array with random numbers
  for (int i = 0; i < N; ++i)
  {
    unc_pos[i].x = xtmp[i]; //random_float();
    unc_pos[i].y = ytmp[i]; //random_float();
  }

  // Allocate memory on the device
  FLOAT2* dev_unc_pos;
  cudaMalloc((void**)&dev_unc_pos, N * sizeof(FLOAT2));

  // Copy data from host to device
  cudaMemcpy(dev_unc_pos, unc_pos, N * sizeof(FLOAT2), cudaMemcpyHostToDevice);


  maxCoord<<<grid, threads, GRID_DIM*sizeof(int)>>>(dev_unc_pos, dev_result, nbead);
  cudaDeviceSynchronize();
  
  cudaMemcpy(result, dev_result, GRID_DIM*sizeof(float), cudaMemcpyDeviceToHost);

  cout << "result[0] = " << result[0] << endl;


  Octree_node root;

  root.set_range(0, nbead);
  root.set_width(2.0f * ceil(result[0]));

  printf("Ranges: (%d, %d)\n", root.points_begin(), root.points_end());
  printf("width: %f\n", root.get_width());
  
  int expected_number_of_nodes = 1000;
  
  Octree_node *all_nodes;
  cudaMalloc((void**)&all_nodes, sizeof(Octree_node) * expected_number_of_nodes);
  cudaMemcpy(all_nodes, &root, sizeof(Octree_node), cudaMemcpyHostToDevice);

  int* dev_buffer1;
  int* dev_buffer2;

  // Allocate memory on the device for dev_buffer1
  cudaMalloc((void**)&dev_buffer1, N * sizeof(int));

  // Allocate memory on the device for dev_buffer2
  cudaMalloc((void**)&dev_buffer2, N * sizeof(int));


  init_nodes<<<grid, threads>>>(dev_buffer1, nbead);
  cudaDeviceSynchronize();
  
  Parameters params(nbead);
  
  int num_nodes_host[1] = {1};

  int* num_nodes;
  cudaMalloc((void**)&num_nodes, sizeof(int));
  cudaMemcpy(num_nodes, num_nodes_host, sizeof(int), cudaMemcpyHostToDevice);


  cudaMemcpy(num_nodes, num_nodes_host, sizeof(int), cudaMemcpyHostToDevice);
  
  // Launch the kernel
  build_octree_kernel<<<1, BLOCK_DIM, 8*sizeof(int)>>>(all_nodes, all_nodes, num_nodes, dev_buffer1, dev_buffer2, dev_unc_pos, params);

  // Synchronize and check for kernel launch errors
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching build_octree_kernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
      // Handle the error, for example, by returning an error code from your function
  }

  // Check for any errors launched by the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "build_octree_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
      // Handle the error
  }

  
  
  //--- for testing ----
  cudaMemcpy(num_nodes_host, num_nodes, sizeof(int), cudaMemcpyDeviceToHost);
  
  cout << "num_nodes_host = " << *num_nodes_host << endl;
  
  Octree_node *host_all_nodes = new Octree_node[expected_number_of_nodes];
  
  cudaError_t cudaMemcpyStatus = cudaMemcpy(host_all_nodes, all_nodes, sizeof(Octree_node) * expected_number_of_nodes, cudaMemcpyDeviceToHost);
  if (cudaMemcpyStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy from device to host failed: %s\n", cudaGetErrorString(cudaMemcpyStatus));
  }


  //Octree_node node1 = host_all_nodes[12];
  //cout << "node1.get_width = " << node1.get_width() << endl;
  
  std::ofstream outFile("bh_tree_data.bin", std::ios::binary);
  if(outFile.is_open())
  {
    // Write N and expected_number_of_nodes
    outFile.write(reinterpret_cast<char*>(&N), sizeof(N));
    outFile.write(reinterpret_cast<char*>(&expected_number_of_nodes), sizeof(expected_number_of_nodes));

    // Write unc_pos
    outFile.write(reinterpret_cast<char*>(unc_pos), sizeof(FLOAT2) * N);

    // Write all_nodes
    outFile.write(reinterpret_cast<char*>(host_all_nodes), sizeof(Octree_node) * expected_number_of_nodes);

    outFile.close();
  }
  else
  {
      std::cerr << "Unable to open file for writing." << std::endl;
  }

  cudaFree(dev_unc_pos);
  cudaFree(dev_buffer1);
  cudaFree(dev_buffer2);
  delete[] unc_pos;
  delete[] host_all_nodes;

}

