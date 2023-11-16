




__global__ void buildTreeKernel(Body* bodies, TreeNode* tree, int* treeSize, int numBodies)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Iterate over all bodies assigned to this thread
    for (int i = idx; i < numBodies; i += stride)
    {
        bool success = false;
        while (!success)
        {
            int cellIndex = findInsertionPoint(bodies[i]);
            int childIndex = getInsertionIndex(tree[cellIndex], bodies[i]);

            // Check if the childIndex is valid and not locked.
            if (childIndex != LOCKED)
            {
                // Attempt to lock the child pointer using atomicCAS (Compare And Swap)
                int expected = NULL_VALUE;
                int desired = LOCKED;
                if (expected == atomicCAS(&tree[cellIndex].child[childIndex], expected, desired))
                {
                    // We have successfully locked the cell.

                    // Check again to make sure the cell is still NULL_VALUE (this can be removed if there's only one potential inserter per cell)
                    if (tree[cellIndex].child[childIndex] == NULL_VALUE)
                    {
                        // Insert body at new index and release lock
                        int newBodyIndex = atomicAdd(treeSize, 1); // Atomically increment tree size to get new body index
                        tree[newBodyIndex].body = bodies[i]; // Copy body to the new index
                        __threadfence(); // Ensure the new body is visible
                        tree[cellIndex].child[childIndex] = newBodyIndex; // Point to the new body index
                    }
                    else
                    {
                        // If the cell is not NULL_VALUE, it means another thread has handled this cell.
                        // We will then have to decide what to do (find a new cell, split the cell, etc.)
                        // This part of the code will depend on the specifics of the tree-building algorithm
                        // For example, we could initialize a new cell and reorganize the bodies as needed.
                        // Here, for simplicity, we just skip the insertion.
                    }
                    success = true; // Flag indicating that insertion succeeded
                }
                else
                {
                    // The cell was locked by another thread. We could:
                    // - Spin-wait until the cell becomes unlocked
                    // - Find a different cell (re-run findInsertionPoint and getInsertionIndex)
                    // For now, we just try again (spin-wait)
                    continue;
                }
            }
            else
            {
                // If childIndex is LOCKED, we need to handle this case (e.g., find another cell).
                // This part of the code depends on your application logic.
                // For now, let's just retry.
                continue;
            }
        }
    }
}

