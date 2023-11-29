
#include <iostream>
#include <cmath>


using namespace std;



struct Cell
{
  int row;
  int col;
  float xcen;
  float ycen;
  int start = -1;
  int end = -1;
};



int getCelliD(float x, float y, float x_min, float y_min, float Wcell, int nSplit)
{
  int col = static_cast<int>((x - x_min) / Wcell);
  int row = static_cast<int>((y - y_min) / Wcell);
  
  return row * nSplit + col;
}



int main()
{


  float x_min = -1.0;
  float y_min = -1.0;

  float maxCoord = 1.0;

  int nSplit = 4; // number of split of each axis!
  int Ncell = nSplit * nSplit; // in 2D !
  
  float W_cell = ceil(2.0 * maxCoord) / nSplit;
  float W_half = W_cell / 2.0f;
  
  Cell *cell = new Cell[Ncell];
  
  int k = 0;
  
  for (int i = 0; i < nSplit; i++)
  {
    for (int j = 0; j < nSplit; j++)
    {
      cell[k].row = i;
      cell[k].col = j;
      
      cell[k].xcen = x_min + j * W_cell + W_half;
      cell[k].ycen = y_min + i * W_cell + W_half;
      
      k++;
    }
  }
  
  
  float x_i = 0.499;
  float y_i = 0.501;
  
  int cell_id = getCelliD(x_i, y_i, x_min, y_min, W_cell, nSplit);
  
  cout << "cell iD = " << cell_id << endl;
  
  
  /*
  for (int i = 0; i < Ncell; i++)
  {
    cout << "cell ID = " << i << "  (xcen, ycen) = " << cell[i].xcen << ", " << cell[i].ycen << endl;
  }
  */
  
  




}



