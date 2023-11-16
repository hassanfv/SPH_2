


const int empty_cell = -1;

//===== coord
struct coord
{
    double x, y, z;
};


//===== get_next_tree_index
int get_next_tree_index()
{
  int old = next_tree_index + 1;
  next_tree_index += 8; //creates space for 8 new cells
  return old;
}


//===== find_subtree_coord
coord *find_subtree_coord(double width, coord *center, int bead_index)
{
  coord *subtree = new coord();
  subtree->x = unc_pos[bead_index].x < center->x ? 0 : 1;
  subtree->y = unc_pos[bead_index].y < center->y ? 0 : 1;
  subtree->z = unc_pos[bead_index].z < center->z ? 0 : 1;
  return subtree;
}


//===== get_subtree_index
int get_subtree_index(coord *sub_block, int base_index)
{
  int tree_child = base_index + (sub_block->x * 4) + (sub_block->y * 2) + sub_block->z;

  return tree_child;
}


//===== update_center_mass
void update_center_mass(int current_node, int bead_index)
{
  octet_center_mass[current_node].x += unc_pos[bead_index].x;
  octet_center_mass[current_node].y += unc_pos[bead_index].y;
  octet_center_mass[current_node].z += unc_pos[bead_index].z;
}


//===== get_udpated_center
coord *get_udpated_center(coord *boxCenter, coord *subtree, double width)
{
  double quarter = width/4.0;
  
  coord *newCenter = new coord();
  
  newCenter->x = subtree->x == 1.0 ? boxCenter->x + quarter : boxCenter->x - quarter;
  newCenter->y = subtree->y == 1.0 ? boxCenter->y + quarter : boxCenter->y - quarter;
  newCenter->z = subtree->z == 1.0 ? boxCenter->z + quarter : boxCenter->z - quarter;
  
  return newCenter;
}


//===== insert_bead_bhtree
void insert_bead_bhtree(int tree_index, int bead_index, coord *boxCenter, double width)
{

  if (indices_bhtree[tree_index] == empty_cell)
  {
    /* Inseart bead with negative index */
    indices_bhtree[tree_index] = -bead_index;
    octet_count_bhtree[tree_index] = 1; // Counts the total number of beads in the cell with index 'tree_index' including beads in its children.

    octet_center_mass[tree_index].x = unc_pos[bead_index].x;
    octet_center_mass[tree_index].y = unc_pos[bead_index].y;
    octet_center_mass[tree_index].z = unc_pos[bead_index].z;
  }
  else if (indices_bhtree[tree_index] < 0)
  {
    /* Index is negative if there’s a node in the cell,
       new positive index will be added for new cell */
    int a_bead_index = -indices_bhtree[tree_index];
    
    indices_bhtree[tree_index] = get_next_tree_index();
    
    octet_count_bhtree[tree_index] = 2;
    
    update_center_mass(tree_index, bead_index);
    
    coord *subtreeA = find_subtree_coord(width, boxCenter, a_bead_index); // return sth like (1, 0, 1)
    coord *subtreeB = find_subtree_coord(width, boxCenter, bead_index);
    
    int subtree_indexA = get_subtree_index(subtreeA, indices_bhtree[tree_index]); // returns 0,1,..., 7 depending on sth like (1, 0, 1) from find_subtree_coord
    int subtree_indexB = get_subtree_index(subtreeB, indices_bhtree[tree_index]);
    
    coord *newCenterA = get_udpated_center(boxCenter, subtreeA, width);
    coord *newCenterB = get_udpated_center(boxCenter, subtreeB, width);
    
    insert_bead_bhtree(subtree_indexA, a_bead_index, newCenterA, width/2.0);
    insert_bead_bhtree(subtree_indexB, bead_index, newCenterB, width/2.0);
    
    delete newCenterA; delete newCenterB;
  } 
  else
  {
    /* Positive index, internal node found */
    octet_count_bhtree[tree_index] = octet_count_bhtree[tree_index]+1;
    
    update_center_mass(tree_index, bead_index);
    coord *subtree = find_subtree_coord(width, boxCenter, bead_index);
    
    int subtree_index = get_subtree_index(subtree, indices_bhtree[tree_index]);
    coord *newCenter = get_udpated_center(boxCenter, subtree, width);
    
    insert_bead_bhtree(subtree_index, bead_index, newCenter, width/2.0);
    delete newCenter;
  }
}


//===== set_initial_width
double set_initial_width()
{
  double width = 0.0;
  double x,y,z;
  
  for (int i = 0; i <= nbead; i++)
  {
    x = fabs(unc_pos[i].x);
    y = fabs(unc_pos[i].y);
    z = fabs(unc_pos[i].z);
    width = x > width ? x : width;
    width = y > width ? y : width;
    width = z > width ? z : width;
  }
  
  rootWidth = ceil(2*width);
}




//===== build_bh_tree
void build_bh_tree()
{
  reinserted++;
  
  /* reset tree */
  std::fill_n(indices_bhtree, 16*nbead, empty_cell);
  
  next_tree_index = 0;
  
  int tree_index = 0;
  set_initial_width();
  
  coord *boxCenter = new coord();
  
  boxCenter->x = 0.0;
  boxCenter->y = 0.0;
  boxCenter->z = 0.0;
  
  for (int i = 1; i <= nbead; i++)
  {
    insert_bead_bhtree(tree_index, i, boxCenter, rootWidth);
  }
  
  /* Tree is marked as rebuilt */
  rebuild = 0;
  
  delete boxCenter;
}








