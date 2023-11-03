

__global__ void build_tree_kernel(float *x, float *y, float *mass, int *count, int *start, int *child, int *index, float *left, float *right, float *bottom, float *top, int n, int m)
{
	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;
	bool newBody = true;

	// build quadtree
	float l; 
	float r; 
	float b; 
	float t;
	int childPath;
	int temp;
	offset = 0;
	while((bodyIndex + offset) < n)
	{
		if(newBody)
		{
			newBody = false;

			l = *left;
			r = *right;
			b = *bottom;
			t = *top;

			temp = 0;
			childPath = 0;
			
			if(x[bodyIndex + offset] < 0.5*(l+r))
			{
				childPath += 1;
				r = 0.5*(l+r);
			}
			else
			{
				l = 0.5*(l+r);
			}
			
			if(y[bodyIndex + offset] < 0.5*(b+t))
			{
				childPath += 2;
				t = 0.5*(t+b);
			}
			else
			{
				b = 0.5*(t+b);
			}
		}
		
		int childIndex = child[temp*4 + childPath];

		// traverse tree until we hit leaf node
		while(childIndex >= n)
		{
			temp = childIndex;
			childPath = 0;
			if(x[bodyIndex + offset] < 0.5*(l+r))
			{
				childPath += 1;
				r = 0.5*(l+r);
			}
			else
			{
				l = 0.5*(l+r);
			}
			if(y[bodyIndex + offset] < 0.5*(b+t))
			{
				childPath += 2;
				t = 0.5*(t+b);
			}
			else
			{
				b = 0.5*(t+b);
			}

			atomicAdd(&x[temp], mass[bodyIndex + offset]*x[bodyIndex + offset]);
			atomicAdd(&y[temp], mass[bodyIndex + offset]*y[bodyIndex + offset]);
			atomicAdd(&mass[temp], mass[bodyIndex + offset]);
			atomicAdd(&count[temp], 1);
			childIndex = child[4*temp + childPath];
		}


		if(childIndex != -2)
		{
			int locked = temp*4 + childPath;
			if(atomicCAS(&child[locked], childIndex, -2) == childIndex)
			{
				if(childIndex == -1)
				{
					child[locked] = bodyIndex + offset;
				}
				else
				{
					//int patch = 2*n;
					int patch = 4*n;
					while(childIndex >= 0 && childIndex < n)
					{
						
				 		int cell = atomicAdd(index,1);
				 		patch = min(patch, cell);
				 		if(patch != cell)
				 		{
				 			child[4*temp + childPath] = cell;
				 		}

            // insert old particle
				 		childPath = 0;
				 		if(x[childIndex] < 0.5*(l+r))
				 		{
				 			childPath += 1;
            }
				 		if(y[childIndex] < 0.5*(b+t))
				 		{
				 			childPath += 2;
				 		}

				 		if(DEBUG)
				 		{
				 			// if(cell >= 2*n){
				 			if(cell >= m)
				 			{
				 				printf("%s\n", "error cell index is too large!!");
				 				printf("cell: %d\n", cell);
				 			}
				 		}
				 		x[cell] += mass[childIndex]*x[childIndex];
				 		y[cell] += mass[childIndex]*y[childIndex];
				 		mass[cell] += mass[childIndex];
				 		count[cell] += count[childIndex];
				 		child[4*cell + childPath] = childIndex;

				 		start[cell] = -1;


            // insert new particle
				 		temp = cell;
				 		childPath = 0;
				 		if(x[bodyIndex + offset] < 0.5*(l+r))
				 		{
				 			childPath += 1;
				 			r = 0.5*(l+r);
				 		}
				 		else
				 		{
				 			l = 0.5*(l+r);
				 		}
				 		if(y[bodyIndex + offset] < 0.5*(b+t))
				 		{
				 			childPath += 2;
				 			t = 0.5*(t+b);
				 		}
				 		else
				 		{
				 			b = 0.5*(t+b);
				 		}
				 		x[cell] += mass[bodyIndex + offset]*x[bodyIndex + offset];
				 		y[cell] += mass[bodyIndex + offset]*y[bodyIndex + offset];
				 		mass[cell] += mass[bodyIndex + offset];
				 		count[cell] += count[bodyIndex + offset];
				 		childIndex = child[4*temp + childPath]; 
				 	}
				
				 	child[4*temp + childPath] = bodyIndex + offset;

				 	__threadfence();  // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

				 	child[locked] = patch;
				}

				// __threadfence(); // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

				offset += stride;
				newBody = true;
			}
		}

		__syncthreads(); // not strictly needed 
	}
}


