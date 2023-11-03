#include <iostream>
#include <iomanip>

const int MAX_PARTICLES = 1000;
const int MAX_CHILDREN = 4;

// Particle structure
struct Particle {
    float x;
    float y;
    float mass;
};

// Quad structure
struct Quad {
    float x1, y1, x2, y2;
    Particle particles[MAX_PARTICLES];
    Quad *children[MAX_CHILDREN];
    int particleIndex = 0;
    int childrenIndex = 0;
    float center_of_mass_x = 0;
    float center_of_mass_y = 0;
    float total_mass = 0;
    int particle_count = 0;
};

bool contains(const Quad& quad, const Particle& particle);
bool insert_to_quad(Quad& quad, const Particle& particle);
void subdivide(Quad& quad);
void traverse_tree(const Quad& quad, int depth = 0);

int main() {
    const int N = 4;
    float xx[] = {-0.90f, -0.80f, -0.40f, -0.20f};
    float yy[] = {0.2f, 0.2f, 0.2f, 0.2f};

    Quad root = {-1, -1, 1, 1, {}, {}, 0, 0, 0, 0};

    for (int i = 0; i < N; i++) {
        Particle p = {xx[i], yy[i], 1.0f};
        insert_to_quad(root, p);
    }

    traverse_tree(root);
    return 0;
}

bool contains(const Quad& quad, const Particle& particle) {
    return (quad.x1 <= particle.x && particle.x < quad.x2) &&
           (quad.y1 <= particle.y && particle.y < quad.y2);
}

bool insert_to_quad(Quad& quad, const Particle& particle) {
    if (!contains(quad, particle)) {
        return false;
    }

    if (quad.childrenIndex == 0 && quad.particle_count < 2) {
        quad.particles[quad.particleIndex++] = particle;
        quad.particle_count++;
        quad.total_mass += particle.mass;
        quad.center_of_mass_x = (quad.center_of_mass_x * (quad.particle_count - 1) + particle.x * particle.mass) / quad.total_mass;
        quad.center_of_mass_y = (quad.center_of_mass_y * (quad.particle_count - 1) + particle.y * particle.mass) / quad.total_mass;
        return true;
    }

    if (quad.childrenIndex == 0 && quad.particle_count == 2) {
        subdivide(quad);
        for (int i = 0; i < quad.particleIndex; i++) {
            for (int j = 0; j < quad.childrenIndex; j++) {
                if (insert_to_quad(*quad.children[j], quad.particles[i])) {
                    break;
                }
            }
        }
        quad.particleIndex = 0;
    }

    for (int i = 0; i < quad.childrenIndex; i++) {
        if (insert_to_quad(*quad.children[i], particle)) {
            quad.particle_count++;
            return true;
        }
    }

    return false;
}

void subdivide(Quad& quad) {
    float hx = (quad.x1 + quad.x2) / 2;
    float hy = (quad.y1 + quad.y2) / 2;

    quad.children[0] = new Quad{quad.x1, quad.y1, hx, hy};
    quad.children[1] = new Quad{hx, quad.y1, quad.x2, hy};
    quad.children[2] = new Quad{quad.x1, hy, hx, quad.y2};
    quad.children[3] = new Quad{hx, hy, quad.x2, quad.y2};
    quad.childrenIndex = 4;
}

void traverse_tree(const Quad& quad, int depth) {
    std::cout << std::setw(2 * depth) << "" << "Quad at depth " << depth << ": x1=" << quad.x1 << ", y1=" << quad.y1 << ", x2=" << quad.x2 << ", y2=" << quad.y2 << std::endl;
    std::cout << std::setw(2 * depth) << "" << "Center of Mass: (" << quad.center_of_mass_x << ", " << quad.center_of_mass_y << ")" << std::endl;
    std::cout << std::setw(2 * depth) << "" << "Particles: ";
    for (int i = 0; i < quad.particleIndex; i++) {
        std::cout << "(" << quad.particles[i].x << ", " << quad.particles[i].y << ") ";
    }
    std::cout << std::endl;

    for (int i = 0; i < quad.childrenIndex; i++) {
        traverse_tree(*quad.children[i], depth + 1);
    }
}

