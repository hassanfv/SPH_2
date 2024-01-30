#include <iostream>
#include <cmath>

// Assuming a maximum number of particles and clumps for simplicity
const int MAX_PARTICLES = 10000;
const int MAX_CLUMPS = 1000;
const int MIN_CLUMP_SIZE = 100; // Minimum number of particles for a clump
double rho_bg = 100.0; // Example value, set according to your simulation

struct Particle {
    double x, y, z; // Position
    double temperature;
    double density;
    bool isCold; // Flag to mark if the particle is cold and dense
};

struct Clump {
    int particleIndices[MIN_CLUMP_SIZE]; // Array to store indices of particles in this clump
    int size; // Number of particles in the clump
};

Particle particles[MAX_PARTICLES]; // Global array of particles
Clump clumps[MAX_CLUMPS]; // Global array of clumps
int numParticles = 0; // Actual number of particles
int numClumps = 0; // Actual number of clumps

double computeDistance(const Particle& a, const Particle& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

void identifyColdParticles() {
    for (int i = 0; i < numParticles; ++i) {
        if (particles[i].temperature < 500 && particles[i].density > 50 * rho_bg) {
            particles[i].isCold = true;
        } else {
            particles[i].isCold = false;
        }
    }
}

void findOrCreateClump(int particleIndex) {
    bool foundClump = false;
    for (int i = 0; i < numClumps; ++i) {
        for (int j = 0; j < clumps[i].size; ++j) {
            if (computeDistance(particles[particleIndex], particles[clumps[i].particleIndices[j]]) < /* Smoothing length calculation here */) {
                clumps[i].particleIndices[clumps[i].size++] = particleIndex;
                foundClump = true;
                break;
            }
        }
        if (foundClump) break;
    }

    if (!foundClump && numClumps < MAX_CLUMPS) {
        clumps[numClumps].particleIndices[0] = particleIndex;
        clumps[numClumps].size = 1;
        numClumps++;
    }
}

void runClumpFinder() {
    identifyColdParticles();
    for (int i = 0; i < numParticles; ++i) {
        if (particles[i].isCold) {
            findOrCreateClump(i);
        }
    }
}

int main() {
    // Initialize particles array with your data

    runClumpFinder();

    // Output clumps for further processing or analysis

    return 0;
}

