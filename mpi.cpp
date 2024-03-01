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


void send_outgoing_parts(list<particle_t>* outgoing_parts, 
                           int send_rank, int destination_rank) {
    // Send the data as an array of MPI_BYTEs to the target process.
    MPI_Send((void*)outgoing_parts->data(), outgoing_parts->size(), PARTICLE, destination_rank, 0, MPI_COMM_WORLD);

    // Clear the outgoing parts
    outgoing_parts->clear();
}

void receive_incoming_parts(list<particle_t>* incoming_parts, int incoming_rank) {
    MPI_Status status;

    // Receive from the process with specified rank
    MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status);

    // Resize your incoming parts buffer based on how much data is
    // being received
    int incoming_parts_size;
    MPI_Get_count(&status, PARTICLE, &incoming_parts_size);
    incoming_parts->resize(incoming_parts_size);
    MPI_Recv((void*)incoming_parts->data(), incoming_parts_size, PARTICLE, incoming_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
}


std::vector<particle_t> incoming_parts;
std::list<particle_t> my_parts;
std::list<particle_t> above_outgoing_parts;
std::list<particle_t> below_outgoing_parts;
std::list<particle_t> ghost_parts;
std::list<particle_t> above_ghost_parts;
std::list<particle_t> below_ghost_parts;
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// // divide the space into num_procs number of squares
	// num_cols = static_cast<int> sqrt(num_procs);
	// num_rows = static_cast<int> (num_procs / num_cols) + (num_procs % num_cols != 0 ? 1 : 0);
	
	// Find particles the current processor need to handle (I am doing 1D layout now)
	double row_width = size / num_procs;
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
        	receive_incoming_walkers(&incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
        	receive_incoming_walkers(&incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        
    } else {
        if (rank + 1 < num_procs) {
        	receive_incoming_walkers(&incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
        	receive_incoming_walkers(&incoming_parts, rank - 1);
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
    	if (rank - 1 >= 0 && abs(parts[particle].y - rank * row_width) <= cutoff) {
    		below_ghost_parts.push_back(particle);
    	}
    	if (rank + 1 < num_procs && abs(parts[particle].y - (rank + 1) * row_width) <= cutoff) {
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
        	receive_incoming_walkers(&incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_walkers(&incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        
    } else {
    	if (rank - 1 >= 0) {
        	receive_incoming_walkers(&incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(particle);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_walkers(&incoming_parts, rank + 1);
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


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
	
	// get a place to receive all particles from all processes in rank 0
	if (rank == 0) {
  		temp_parts = malloc(sizeof(particle_t) * num_parts);
	}
	// gather to root processor
    MPI_Gather(&temp_parts, my_parts.size(), PARTICLE, my_parts, my_parts.size(), PARTICLE, 0, MPI_COMM_WORLD);
    // in root processor, sort the particles
    if (rank == 0) {
    	particle_t* parts = new particle_t[num_parts];
    	for (int i = 0; i < num_parts; ++i) {
    		int id = temp_parts[i].id;
    		parts[id - 1] = temp_parts[i];
    	}
    	free(temp_parts);
    }
}



