#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <list>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void send_outgoing_parts(std::list<particle_t>* outgoing_parts, int send_rank, int destination_rank) {
    // Get the size of the list
    int num_particles = outgoing_parts->size();

    // Send the number of particles first
    MPI_Send(&num_particles, 1, MPI_INT, destination_rank, 0, MPI_COMM_WORLD);

    // Send each particle one by one
    for (const auto& particle : *outgoing_parts) {
        MPI_Send(&particle, sizeof(particle_t), MPI_BYTE, destination_rank, 0, MPI_COMM_WORLD);
    }

    // Clear the outgoing parts
    outgoing_parts->clear();
}



void receive_incoming_parts(std::list<particle_t>& incoming_parts, int incoming_rank) {
    MPI_Status status;

    // Receive the number of particles
    int num_particles;
    MPI_Recv(&num_particles, 1, MPI_INT, incoming_rank, 0, MPI_COMM_WORLD, &status);

    // Receive each particle one by one and add to the list
    for (int i = 0; i < num_particles; ++i) {
        particle_t particle;
        MPI_Recv(&particle, sizeof(particle_t), MPI_BYTE, incoming_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        incoming_parts.push_back(particle);
    }
}



std::list<particle_t> incoming_parts;
std::list<particle_t> my_parts;
std::list<particle_t> above_outgoing_parts;
std::list<particle_t> below_outgoing_parts;
std::list<particle_t> ghost_parts;
std::list<particle_t> above_ghost_parts;
std::list<particle_t> below_ghost_parts;
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	double row_width = size / num_procs;
	// Find particles the current processor need to handle (1D layout)
	for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
        int bin = static_cast<int>(parts[i].y / row_width);
        if (bin == rank){
        	my_parts.push_back(parts[i]);
        }
        if (bin == rank - 1 && abs(parts[i].y - rank * row_width) <= cutoff) {
    		ghost_parts.push_back(parts[i]);
    	}
    	if (bin == rank + 1 && abs(parts[i].y - (rank + 1) * row_width) <= cutoff) {
    		ghost_parts.push_back(parts[i]);
    	}    
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_width = size / num_procs;
	// apply force to all particles in my_parts
    for (particle_t& particle : my_parts) {
    	for (particle_t& neighbor : my_parts) {
    		apply_force(particle, neighbor);
    	}
    	for (particle_t& neighbor : ghost_parts) {
    		apply_force(particle, neighbor);
    	}
    }
    ghost_parts.clear();
    // move particles and find outgoing particles
    for (particle_t& particle : my_parts) {
    	move(particle, size);
    	int bin = static_cast<int>(particle.y / row_width);
    	if (bin == rank - 1) {
    		below_outgoing_parts.push_back(particle);
    		my_parts.remove(particle);
    	}
    	if (bin == rank + 1) {
    		above_outgoing_parts.push_back(particle);
    		my_parts.remove(particle);
    	}
    	particle.ax = particle.ay = 0;
    }
    // redistribute parts, split by even / odd to avoid dead lock
    if (rank % 2 == 0) {
    	if (rank + 1 < num_procs) {
    		send_outgoing_parts(&above_outgoing_parts, rank, rank + 1);
    	}
        if (rank - 1 >= 0) {
        	send_outgoing_parts(&below_outgoing_parts, rank, rank - 1);
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        
    } else {
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
    		send_outgoing_parts(&above_outgoing_parts, rank, rank + 1);
    	}
        if (rank - 1 >= 0) {
        	send_outgoing_parts(&below_outgoing_parts, rank, rank - 1);
        }
    }
    // send ghost parts, split by even / odd to avoid dead lock
    for (particle_t& particle : my_parts) {
    	if (rank - 1 >= 0 && abs(particle.y - rank * row_width) <= cutoff) {
    		below_ghost_parts.push_back(particle);
    	}
    	if (rank + 1 < num_procs && abs(particle.y - (rank + 1) * row_width) <= cutoff) {
    		above_ghost_parts.push_back(particle);
    	}
    }
    if (rank % 2 == 0) {
    	if (rank + 1 < num_procs) {
    		send_outgoing_parts(&above_ghost_parts, rank, rank + 1);
    	}
        if (rank - 1 >= 0) {
        	send_outgoing_parts(&below_ghost_parts, rank, rank - 1);
        }
        if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        
    } else {
    	if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
    		send_outgoing_parts(&above_ghost_parts, rank, rank + 1);
    	}
        if (rank - 1 >= 0) {
        	send_outgoing_parts(&below_ghost_parts, rank, rank - 1);
        }
    }
}


#include <algorithm> // for std::sort

void gather_for_save(particle_t* my_parts, int num_parts, double size, int rank, int num_procs) {
    particle_t* temp_parts = nullptr;

    // Allocate memory on the root process
    if (rank == 0) {
        temp_parts = new particle_t[num_parts * num_procs];
    }

    // Gather particles to the root process
    MPI_Gather(my_parts, num_parts, PARTICLE, temp_parts, num_parts, PARTICLE, 0, MPI_COMM_WORLD);

    // Sort the particles on the root process
    if (rank == 0) {
        // Sort temp_parts by particle id
        std::sort(temp_parts, temp_parts + (num_parts * num_procs),
                  [](const particle_t& a, const particle_t& b) {
                      return a.id < b.id;
                  });

        // Remember to free the allocated memory
        delete[] temp_parts;
    }
}

void cleanup() {
    my_parts.clear();
    above_outgoing_parts.clear();
    below_outgoing_parts.clear();
    ghost_parts.clear();
    above_ghost_parts.clear();
    below_ghost_parts.clear();
    incoming_parts.clear();
}


