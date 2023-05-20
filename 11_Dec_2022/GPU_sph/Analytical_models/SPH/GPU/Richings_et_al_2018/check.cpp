
#include <iostream>
#include <random>

using namespace std;

int main() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < 10; i++)
    {
    float random_number = dis(gen);

    cout << "Random number between -1 and 1: " << random_number << endl;
    }

    return 0;
}

