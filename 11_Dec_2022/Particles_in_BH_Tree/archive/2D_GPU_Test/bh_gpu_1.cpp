




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
__device__ void count_points(const int *in_points, FLOAT3* dev_unc_pos, int *smem, int range_begin, int range_end, FLOAT3 center)
{

  if(threadIdx.x < 8) smem[threadIdx.x] = 0;

  __syncthreads();

  for(int iter=range_begin+threadIdx.x; iter<range_end; iter+=blockDim.x)
  {
    FLOAT3 p = dev_unc_pos[in_points[iter]];
    int x = p.x < center.x ? 0 : 1;
    int y = p.y < center.y ? 0 : 1;
    int z = p.z < center.z ? 0 : 1;
    int i = x*4 + y*2 + z;
    atomicAdd(&smem[i], 1);
  }
  __syncthreads();
}


//===== scan_offsets
__device__ void scan_offsets(int node_points_begin, int* smem)
{

  int *smem2 = &smem[8];
  
  if(threadIdx.x == 0)
  {
    for(int i = 0; i < 8; i++)
    {
      smem2[i] = i == 0 ? 0 : smem2[i-1] + smem[i-1];
    }
    for (int i = 0; i < 8; i++)
    {
      smem2[i] += node_points_begin;
    }
  }
  __syncthreads();
}


//===== reorder_points
__device__ void reorder_points(int *out_points, const int *in_points, FLOAT3 *dev_unc_pos, int *smem, int range_begin, int range_end, FLOAT3 center)
{

  int *smem2 = &smem[8];
  
  for(int iter = range_begin+threadIdx.x; iter<range_end; iter+=blockDim.x)
  {
    FLOAT3 p = dev_unc_pos[in_points[iter]];

    int x = p.x < center.x ? 0 : 1;
    int y = p.y < center.y ? 0 : 1;
    int z = p.z < center.z ? 0 : 1;
    
    int i = x*4 + y*2 + z;
    
    int dest = atomicAdd(&smem2[i], 1);
    
    out_points[dest] = in_points[iter];
  }
  __syncthreads();
}


//===== prepare_children
__device__ void prepare_children(Octree_node *children, Octree_node &node, int *smem)
{

  const FLOAT3 center = node.center();
  
  float half = node.width() / 2.0f;
  float quarter = half / 2.0f;
  
  smem[7] = node.points_begin();
  
  for(int i = 0; i < 8; i++)
  {
    int xf, yf, zf;

    zf = i % 2;
    if (zf == 0) zf = -1;
    
    yf = (i / 2) % 2;
    if (yf == 0) yf = -1;
    
    xf = i / 4;
    if (xf == 0) xf = -1;
    
    children[i].set_center(center.x + quarter * xf, center.y + quarter * yf, center.z + quarter * zf);
    children[i].set_width(half);
    children[i].set_range(smem[7+i], smem[8+i]);
  }
}


//===== build_octree_kernel
__global__ void build_octree_kernel(Octree_node *nodes, Octree_node *all_nodes, int *num_nodes, int *points1, int *points2, FLOAT3 *dev_unc_pos, Parameters params)
{

  __shared__ int smem[16];
  
  Octree_node &node = nodes[blockIdx.x];
  
  int num_points = node.num_points();
  bool exit = check_points(node, points1, points2, num_points, params);
  
  if(exit) return;
  
  float3 center = node.center();
  
  int range_begin = node.points_begin();
  int range_end = node.points_end();
  
  const int *in_points = params.point_selector == 0 ? points1 : points2;
  
  int *out_points = params.point_selector == 0 ? points2 : points1;
  
  count_points(in_points, dev_unc_pos, smem, range_begin, range_end, center);
  
  scan_offsets(node.points_begin(), smem);
  
  reorder_points(out_points, in_points, dev_unc_pos, smem, range_begin, range_end, center);
  
  if(threadIdx.x == blockDim.x-1)
  {
    int next_node = atomicAdd(num_nodes, 8);
    
    node.set_children(next_node);
    
    Octree_node *children = &all_nodes[next_node];
    
    prepare_children(children, node, smem);
    
    build_octree_kernel<<<8, blockDim.x, 16*sizeof(int)>>>(children, all_nodes, num_nodes, points1, points2, dev_unc_pos, Parameters(params, true));
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


//===== calculate_centers_mass
__global__ void calculate_centers_mass(Octree_node *all_nodes, int *num_nodes, int *points, FLOAT3 *dev_unc_pos)
{

  FLOAT x = 0.0, y = 0.0, z = 0.0;
  
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
      z += dev_unc_pos[points[i]].z;
    }
    
    x /= node.num_points();
    y /= node.num_points();
    z /= node.num_points();
    
    node.set_center_mass(x/node.num_points(), y/node.num_points(), z/node.num_points());
  }
}



void build_bh_tree()
{

  dim3 threads(BLOCK_DIM, 1, 1);
  
  dim3 grid((int) ceil((nbead + 1.0) / (float) threads.x), 1, 1);
  
  FLOAT *dev_result;
  
  FLOAT result[BLOCK_DIM];
  
  cudaMalloc((void **)&dev_result, BLOCK_DIM*sizeof(FLOAT));
  
  maxCoord<<<grid, threads, BLOCK_DIM*sizeof(FLOAT)>>>(dev_unc_pos, dev_result, nbead);
  
  cudaMemcpy(result, dev_result, BLOCK_DIM*sizeof(FLOAT), cudaMemcpyDeviceToHost);
  
  Octree_node root;
  
  root.set_range(0, nbead);
  
  root.set_width(2*ceil(result[0]));
  
  cudaMemcpy(all_nodes, &root, sizeof(Octree_node), cudaMemcpyHostToDevice);
  
  init_nodes<<<grid, threads>>>(dev_buffer1, nbead);
  
  Parameters params(nbead);
  
  num_nodes_host[0] = 1;
  
  cudaMemcpy(num_nodes, num_nodes_host, sizeof(int), cudaMemcpyHostToDevice);
  
  build_octree_kernel<<<1, BLOCK_DIM, shared_mem>>>(all_nodes, all_nodes, num_nodes, dev_buffer1, dev_buffer2, dev_unc_pos, params);
  
  calculate_centers_mass<<<grid,threads>>>(all_nodes, num_nodes, dev_buffer1, dev_unc_pos);
  
  cudaMemcpy(num_nodes_host, num_nodes, sizeof(int), cudaMemcpyDeviceToHost);
}





