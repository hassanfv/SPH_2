


struct Parameters {
    int depth;          // Current depth in the octree
    int max_depth;      // Maximum allowable depth for the octree
    int min_points;     // Minimum number of points required to split a node further
    int point_selector; // Selector to determine which points buffer to use

    // Constructor
    __host__ __device__ Parameters(int nbead) {
        // Initialize the parameters with appropriate values
        // The actual initialization will depend on the specific application logic.
        // For example:
        depth = 0; // Starting depth could be 0
        max_depth = calculate_max_depth_based_on_nbead(nbead);
        min_points = calculate_min_points_based_on_nbead(nbead);
        point_selector = 0; // Initial value for point selector
    }

    // Overloaded constructor for creating a new parameters object for deeper levels
    __host__ __device__ Parameters(const Parameters& other, bool incrementDepth) {
        // Copy all parameters from 'other'
        depth = other.depth;
        max_depth = other.max_depth;
        min_points = other.min_points;
        point_selector = other.point_selector;

        // Increment depth if required
        if (incrementDepth) {
            depth++;
        }

        // Change point_selector or other properties if necessary
    }

private:
    // Helper methods to calculate values based on 'nbead'
    __host__ __device__ int calculate_max_depth_based_on_nbead(int nbead) {
        // Example calculation (actual logic may vary)
        return (int)log2(nbead);
    }

    __host__ __device__ int calculate_min_points_based_on_nbead(int nbead) {
        // Example calculation (actual logic may vary)
        return nbead / 10;
    }
};



