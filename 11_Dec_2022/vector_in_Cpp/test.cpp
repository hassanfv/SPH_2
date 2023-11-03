
#include <iostream>
#include <vector>

using namespace std;

int main()
{

  vector<float> x = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  x.push_back(60.0f);
  x.push_back(70.0f);
  x.push_back(80.0f);

  for (int i = 0; i < x.size(); i++)
  {
  
    cout << "x[" << i << "] = " << x[i] << endl;    
     
  }
  

  return 0;
}
