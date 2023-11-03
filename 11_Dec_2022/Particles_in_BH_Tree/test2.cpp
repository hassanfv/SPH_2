#include <iostream>
#include <cmath>


using namespace std;

int next_tree_index = 0;

//===== coord
struct coord
{
  float x, y, z;
};


//===== set_initial_width
float set_initial_width(coord *unc_pos, int nbead)
{

  float width = 0.0f;
  float x, y, z;
  
  for (int i = 0; i < nbead; i++)
  {
    x = fabs(unc_pos[i].x);
    y = fabs(unc_pos[i].y);
    
    width = x > width ? x : width;
    width = y > width ? y : width;
  }
  
  return ceil(2.0f * width);
}


//===== find_subtree_coord
coord *find_subtree_coord(coord *unc_pos, coord *center, int bead_index)
{

  coord *subtree = new coord();
  
  subtree->x = unc_pos[bead_index-1].x < center->x ? 0 : 1;
  subtree->y = unc_pos[bead_index-1].y < center->y ? 0 : 1;
  
  return subtree;
}


//===== get_subtree_index
int get_subtree_index(coord *sub_block, int base_index)
{
  int tree_child = base_index + (sub_block->x * 2) + sub_block->y;
  
  return tree_child;
}


//===== get_updated_center
coord *get_updated_center(coord *boxCenter, coord *subtree, float width)
{
  float quarter = width / 4.0f;
  
  coord *newCenter = new coord();
  
  newCenter->x = subtree->x == 1 ? boxCenter->x + quarter : boxCenter->x - quarter;
  newCenter->y = subtree->y == 1 ? boxCenter->y + quarter : boxCenter->y - quarter;
  
  return newCenter;
}


//===== get_next_tree_index
int get_next_tree_index()
{
  int old = next_tree_index + 1;
  next_tree_index += 4;
  
  return old;
}


//===== insert_bead_bhtree
void insert_bead_bhtree(int *indices_bhtree, int *octet_count_bhtree, int tree_index, int bead_index, coord *boxCenter, coord *unc_pos,
                        float width, int empty_cell)
{
  if (indices_bhtree[tree_index] == empty_cell)
  {
    // Insert bead with negative index
    indices_bhtree[tree_index] = -bead_index;
    octet_count_bhtree[tree_index] = 1;
  }
  
  else if (indices_bhtree[tree_index] < 0)
  
  {
    int a_bead_index = -indices_bhtree[tree_index];
    
    indices_bhtree[tree_index] = get_next_tree_index();
    octet_count_bhtree[tree_index] = 2;
    
    coord *subtreeA = find_subtree_coord(unc_pos, boxCenter, a_bead_index);
    coord *subtreeB = find_subtree_coord(unc_pos, boxCenter, bead_index);
    
    cout << "subtreeA = " << subtreeA->x << ", " << subtreeA->y << endl;
    cout << "unc_pos.x, y = " << unc_pos[a_bead_index].x << ", " << unc_pos[a_bead_index].x << endl;
  
    int subtree_indexA = get_subtree_index(subtreeA, indices_bhtree[tree_index]);
    int subtree_indexB = get_subtree_index(subtreeB, indices_bhtree[tree_index]);
    
    cout << "subtree_indexA, subtree_indexB = " << subtree_indexA << ", " << subtree_indexB << endl;
    
    coord *newCenterA = get_updated_center(boxCenter, subtreeA, width);
    coord *newCenterB = get_updated_center(boxCenter, subtreeB, width);
  
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, subtree_indexA, a_bead_index, newCenterA, unc_pos, width/2.0, empty_cell);
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, subtree_indexB, bead_index, newCenterB, unc_pos, width/2.0, empty_cell);
  }
  
  else
  {
    // positive index; internal node found.
    octet_count_bhtree[tree_index] = octet_count_bhtree[tree_index] + 1;
    
    coord *subtree = find_subtree_coord(unc_pos, boxCenter, bead_index);
    
    int subtree_index = get_subtree_index(subtree, indices_bhtree[tree_index]);
    
    coord *newCenter = get_updated_center(boxCenter, subtree, width);
    
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, subtree_index, bead_index, newCenter, unc_pos, width/2.0, empty_cell);

    delete newCenter;    
  }

}





int main()
{
  
  float rootWidth = 0.0f;

  int tree_index = 0;

  //------
  //float xtmp[] = {0.1, 0.2, 0.3, 0.4, 0.5};
  //float ytmp[] = {0.7, 0.7, 0.7, 0.7, 0.7};
  float xtmp[] = {0.2, 0.7, 0.6, 0.8};
  float ytmp[] = {0.6, 0.7, 0.2, 0.4};
  
  const int nbead = sizeof(xtmp) / sizeof(xtmp[0]);
  
  cout << "nbead = " << nbead << endl;

  coord *unc_pos = new coord[nbead];

  for (int i = 0; i < nbead; i++)
  {
    unc_pos[i].x = xtmp[i];
    unc_pos[i].y = ytmp[i];
  }
  //------

  for (int j = 0; j < nbead; j++)
  {
    cout << "x, y = " << unc_pos[j].x << ", " << unc_pos[j].y << endl;
  }
  
  
  coord *boxCenter = new coord();
  boxCenter->x = 0.5f;
  boxCenter->y = 0.5f;
  
  
  rootWidth = set_initial_width(unc_pos, nbead);
  
  cout << endl;
  cout << "rootWidth = " << rootWidth << endl;

  cout << endl;
  
  for (int i = 0; i < nbead; i++)
  {
  coord *subtree = new coord();
  int tree_child = 0;
  
  subtree = find_subtree_coord(unc_pos, boxCenter, i);
  tree_child = get_subtree_index(subtree, tree_index);
  
  cout << "subtree = " << subtree->x << ", " << subtree->y << ",  subtree index = " << tree_child << endl;
  
  delete[] subtree;
  }
  
  cout << endl;
  
  
  int *indices_bhtree = new int[8*nbead];
  int empty_cell = 0; // This implies empty_cell!
  /* reset tree */
  std::fill_n(indices_bhtree, 8*nbead, empty_cell);
  
  int *octet_count_bhtree = new int[8*nbead];
  
 
  for (int i = 1; i <= nbead; i++)
  {
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, tree_index, i, boxCenter, unc_pos, rootWidth, empty_cell);
  }
  
  
  
  for (int i = 0; i < 8*nbead; i++)
  {
    cout << indices_bhtree[i] << ", ";
  }
  cout << endl;
  
  
  delete[] indices_bhtree;
  delete[] octet_count_bhtree;
  
}








