Vector rij = {b[j].position.x - b[i].position.x, b[j].position.y - b[i].position.y};
double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
double f = (GRAVITY * b[j].mass) / (r * r * r + (E * E));
Vector force = {rij.x * f, rij.y * f};
