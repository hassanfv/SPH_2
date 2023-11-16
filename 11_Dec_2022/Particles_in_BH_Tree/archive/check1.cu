



__global__ void build_octree_kernel(Octree_node *nodes, Octree_node *all_nodes, int *num_nodes, int *points1, int *points2, FLOAT3 *dev_unc_pos, Parameters params)
{
  __shared__ int smem[16]; // Shared memory for communication within the block.

  Octree_node &node = nodes[blockIdx.x]; // Take the node corresponding to this block.
  
  int num_points = node.num_points(); // Assume 16 points for the root node.
  bool exit = check_points(node, points1, points2, num_points, params); // Check if we should exit.
  
  if(exit) return; // If the depth is maxed or not enough points, we stop.
  
  float3 center = node.center(); // Center of the node.
  
  int range_begin = node.points_begin(); // Beginning index of the points in this node.
  int range_end = node.points_end(); // Ending index of the points in this node.

  const int* in_points = params.point_selector == 0 ? points1 : points2; // Input points buffer.
  
  int *out_points = params.point_selector == 0 ? points2 : points1; // Output points buffer.
  
  count_points(in_points, dev_unc_pos, smem, range_begin, range_end, center); // Count points in each octant.
  
  scan_offsets(node.points_begin(), smem); // Compute exclusive scan to determine output positions.
  
  reorder_points(out_points, in_points, dev_unc_pos, smem, range_begin, range_end, center); // Reorder points into octants.
  
  if(threadIdx.x == blockDim.x-1) // Only the last thread of the block does the following.
  {
    int next_node = atomicAdd(num_nodes, 8); // Allocate space for 8 children nodes.
    
    node.set_children(next_node); // Set the children index for the current node.
    
    Octree_node *children = &all_nodes[next_node]; // Pointer to the children.
    
    prepare_children(children, node, smem); // Set up the children nodes.
    
    // Recursively call this kernel for each of the 8 children.
    build_octree_kernel<<<8, blockDim.x, 16*sizeof(int)>>>(children, all_nodes, num_nodes, points1, points2, dev_unc_pos, Parameters(params, true));
  }
}

