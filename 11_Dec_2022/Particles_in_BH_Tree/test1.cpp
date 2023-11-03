#include <iostream>
#include <cmath>


using namespace std;


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
  
  subtree->x = unc_pos[bead_index].x < center->x ? 0 : 1;
  subtree->y = unc_pos[bead_index].y < center->y ? 0 : 1;
  
  return subtree;
}


//===== get_subtree_index
int get_subtree_index(coord *sub_block, int base_index)
{
  int tree_child = base_index + (sub_block->x * 2) + sub_block->y;
  
  return tree_child;
}



int main()
{
  
  float initialWidth = 0.0f;

  int tree_index = 0;

  //------
  const int nbead = 2;
  //float xtmp[] = {0.1, 0.2, 0.3, 0.4, 0.5};
  //float ytmp[] = {0.7, 0.7, 0.7, 0.7, 0.7};
  float xtmp[] = {0.2, 0.7};
  float ytmp[] = {0.6, 0.1};

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
  
  
  initialWidth = set_initial_width(unc_pos, nbead);
  
  cout << endl;
  cout << "initialWidth = " << initialWidth << endl;

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
  
  
  

}








