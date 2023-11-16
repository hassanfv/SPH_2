
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
