
#include <iostream>
#include <random>

using namespace std;

int main() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    int n = 0;

    while (n < 100)
    {
    
    float x = dis(gen);
    
    n++;

    if (x < 0.0f)
        continue;
    
    cout << "n, x = " << n << ", " << x << endl;
    

    }

    return 0;
}

