#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <vector>
#include <tuple>


using namespace std;

//const int nbead = 4;

//int nbead;

float xtmp[] = {-0.9, 0.2, 0.9};//, 0.1, 0.9};
float ytmp[] = {0.5, 0.2, 0.9};//, -0.9, -0.1};
float mass_tmp[] = {2.0, 4.0, 6.0};

int nbead = sizeof(xtmp) / sizeof(xtmp[0]);

int next_tree_index = 0;

// Global counter for box_id
int box_id_counter = 0;

// Global vector to store boxes' data
vector<tuple<int, float, float, float>> boxes_data;

//===== coord
struct coord
{
  float x, y, z;
};


float *octet_mass = new float[8*nbead];


//=== defining octet_center_mass GLOBALLY !!!
coord *octet_center_mass = new coord[8*nbead];

coord *unc_pos = new coord[nbead];

float *mass = new float[nbead];

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
  
  subtree->x = unc_pos[bead_index - 1].x < center->x ? 0 : 1;
  subtree->y = unc_pos[bead_index - 1].y < center->y ? 0 : 1;
  
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



//===== update_center_mass
void update_center_mass(int tree_index, int bead_index)
{

  float total_mass = octet_mass[tree_index] + mass[bead_index - 1];
  
  octet_center_mass[tree_index].x = (octet_center_mass[tree_index].x * octet_mass[tree_index] + unc_pos[bead_index - 1].x * mass[bead_index - 1]) / total_mass;
  octet_center_mass[tree_index].y = (octet_center_mass[tree_index].y * octet_mass[tree_index] + unc_pos[bead_index - 1].y * mass[bead_index - 1]) / total_mass;
  
  octet_mass[tree_index] = total_mass;
  
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
    
    octet_center_mass[tree_index].x = unc_pos[bead_index - 1].x;
    octet_center_mass[tree_index].y = unc_pos[bead_index - 1].y;
    
    octet_mass[tree_index] = mass[bead_index - 1];
  }
  
  else if (indices_bhtree[tree_index] < 0)
  
  {
    int a_bead_index = -indices_bhtree[tree_index];
    
    indices_bhtree[tree_index] = get_next_tree_index();
    
    octet_count_bhtree[tree_index] = 2;
    
    update_center_mass(tree_index, bead_index);
    
    coord *subtreeA = find_subtree_coord(unc_pos, boxCenter, a_bead_index);
    coord *subtreeB = find_subtree_coord(unc_pos, boxCenter, bead_index);
      
    int subtree_indexA = get_subtree_index(subtreeA, indices_bhtree[tree_index]);
    int subtree_indexB = get_subtree_index(subtreeB, indices_bhtree[tree_index]);
    
    coord *newCenterA = get_updated_center(boxCenter, subtreeA, width);
    coord *newCenterB = get_updated_center(boxCenter, subtreeB, width);
    
    //---------- Store the box data for newCenterA and newCenterB
    ++box_id_counter;
    boxes_data.push_back(make_tuple(box_id_counter, newCenterA->x, newCenterA->y, width/2.0));

    ++box_id_counter;
    boxes_data.push_back(make_tuple(box_id_counter, newCenterB->x, newCenterB->y, width/2.0));
    //----------
  
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, subtree_indexA, a_bead_index, newCenterA, unc_pos, width/2.0, empty_cell);
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, subtree_indexB, bead_index, newCenterB, unc_pos, width/2.0, empty_cell);
  }
  
  else
  {
    // positive index; internal node found.
    octet_count_bhtree[tree_index] = octet_count_bhtree[tree_index] + 1;
    
    update_center_mass(tree_index, bead_index);
    
    coord *subtree = find_subtree_coord(unc_pos, boxCenter, bead_index);
    
    int subtree_index = get_subtree_index(subtree, indices_bhtree[tree_index]);
    
    coord *newCenter = get_updated_center(boxCenter, subtree, width);
    
    ++box_id_counter;
    boxes_data.push_back(make_tuple(box_id_counter, newCenter->x, newCenter->y, width/2.0));
    
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, subtree_index, bead_index, newCenter, unc_pos, width/2.0, empty_cell);

    delete newCenter;    
  }

}





int main()
{
  
  float rootWidth = 0.0f;

  int tree_index = 0;
  
  // Set-up random number generator
  mt19937 gen(42);
  uniform_real_distribution<> dis(-1.0, 1.0);
  
  for (int i = 0; i < nbead; i++)
  {
    unc_pos[i].x = xtmp[i];//]dis(gen);
    unc_pos[i].y = ytmp[i];//dis(gen);
    mass[i] = mass_tmp[i];
  }
  
  cout << "nbead = " << nbead << endl;
  
  for (int i = 0; i < 8*nbead; i++)
  {
    octet_center_mass[i].x = 0.0f;
    octet_center_mass[i].y = 0.0f;
  }
  
  
  cout << "octet_center_mass XXX = [";
  for (int i = 0; i < 8*nbead; i++)
  {
    cout << octet_center_mass[i].x << ", ";
  }
  cout << "]" << endl;
  cout << endl;
  

  for (int j = 0; j < nbead; j++)
  {
    cout << "x, y = " << unc_pos[j].x << ", " << unc_pos[j].y << endl;
  }
  
  
  coord *boxCenter = new coord();
  boxCenter->x = 0.0f;
  boxCenter->y = 0.0f;
  
  
  rootWidth = set_initial_width(unc_pos, nbead);
  
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
  
  for (int i = 0; i < 8*nbead; i++)
  {
    octet_count_bhtree[i] = 0;
    octet_mass[i] = 0.0f;
  }
  
 
  //==== Main BHtree Construction loop!
  for (int i = 1; i <= nbead; i++)
  {
    insert_bead_bhtree(indices_bhtree, octet_count_bhtree, tree_index, i, boxCenter, unc_pos, rootWidth, empty_cell);
  }
  
  
  cout << endl;
  cout << endl;
  cout << "indices_bhtree = [";
  for (int i = 0; i < 8*nbead; i++)
  {
    cout << indices_bhtree[i] << ", ";
  }
  cout << "]" << endl;
  cout << endl;
  
  
  cout << "octet_count_bhtree = [";
  for (int i = 0; i < 8*nbead; i++)
  {
    cout << octet_count_bhtree[i] << ", ";
  }
  cout << "]" << endl;
  cout << endl;
  
  
  cout << "octet_center_mass X = [";
  for (int i = 0; i < 8*nbead; i++)
  {
    cout << octet_center_mass[i].x << ", ";
  }
  cout << "]" << endl;
  cout << endl;
  
  cout << "octet_center_mass Y = [";
  for (int i = 0; i < 8*nbead; i++)
  {
    cout << octet_center_mass[i].y << ", ";
  }
  cout << "]" << endl;
  cout << endl;
  
  cout << "octet_mass = [";
  for (int i = 0; i < 8*nbead; i++)
  {
    cout << octet_mass[i] << ", ";
  }
  cout << "]" << endl;
  cout << endl;
  
  
  
  // Save to boxes.csv
  ofstream outfile("boxes.csv");
  outfile << "box_id,center_x,center_y,width\n";
  for (const auto& box : boxes_data) {
      outfile << get<0>(box) << "," << get<1>(box) << "," << get<2>(box) << "," << get<3>(box) << "\n";
  }
  outfile.close();
  
  
  // Save to particles.csv
  ofstream particles_file("particles.csv");
  particles_file << "particle_id,x,y\n";
  for (int i = 0; i < nbead; i++)
  {
      particles_file << i + 1 << "," << unc_pos[i].x << "," << unc_pos[i].y << "\n";
  }
  particles_file.close();
  
  delete[] indices_bhtree;
  delete[] octet_count_bhtree;
  
}








