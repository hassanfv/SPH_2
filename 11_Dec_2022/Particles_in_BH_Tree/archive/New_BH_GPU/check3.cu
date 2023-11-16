



#include <cuda.h>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#define NULL_VALUE -1
#define MAX_BODIES 1024

struct Body {
    float3 position;
    // ... other properties
};

struct TreeNode {
    int child[8];
    Body body;
    // ... other properties
};

__device__ int findInsertionPoint(Body body) {
    // This function should return the index of the node where the body should be inserted
    // You'll need to define how to calculate this based on your octree structure
    // For now, let's assume we are just inserting into the root
    return 0;
}

__device__ int getInsertionIndex(TreeNode& node, Body body) {
    // This function should return the index of the child cell where the body should be inserted
    // You'll need to calculate this based on the body's position and the node's bounds
    // Dummy implementation for illustration
    return static_cast<int>((body.position.x + body.position.y + body.position.z) / 3) % 8;
}

__global__ void buildTreeKernel(Body* bodies, TreeNode* tree, int* treeSize, int numBodies) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numBodies; i += stride) {
        bool inserted = false;
        while (!inserted) {
            int cellIndex = findInsertionPoint(bodies[i]);
            int childIndex = getInsertionIndex(tree[cellIndex], bodies[i]);

            // Using atomicExch for locking
            int expected = NULL_VALUE;
            if (atomicExch(&tree[cellIndex].child[childIndex], LOCKED) == expected) {
                // Lock acquired
                
                if (tree[cellIndex].child[childIndex] == NULL_VALUE) {
                    // If cell is empty, insert the body
                    tree[cellIndex].child[childIndex] = i; // Assuming body index is sufficient
                    inserted = true;
                } else {
                    // Cell is not empty, need to create a new cell
                    int newCellIndex = atomicAdd(treeSize, 1);
                    initializeTreeNode(&tree[newCellIndex]); // You need to define how a TreeNode is initialized

                    // Move the existing body to the new cell
                    int existingBodyIdx = tree[cellIndex].child[childIndex];
                    int newChildIndex = getInsertionIndex(tree[newCellIndex], bodies[existingBodyIdx]);
                    tree[newCellIndex].child[newChildIndex] = existingBodyIdx;

                    // Insert the new body into the new cell
                    childIndex = getInsertionIndex(tree[newCellIndex], bodies[i]);
                    tree[newCellIndex].child[childIndex] = i;

                    // Ensure visibility of changes
                    __threadfence();

                    // Update the parent's pointer to point to the new cell
                    tree[cellIndex].child[childIndex] = newCellIndex;

                    inserted = true;
                }

                // Unlocking
                atomicExch(&tree[cellIndex].child[childIndex], expected);
            }
            // Note: No __syncthreads() here because we are dealing with global memory and threads across blocks cannot be synchronized this way.
        }
    }
}

// You will also need to define initializeTreeNode() and any other functions used here.

