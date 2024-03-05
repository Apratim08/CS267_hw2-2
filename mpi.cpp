#include <limits> // Include the <limits> header
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <list>
#include <cstdlib> // for exit()

bool particle_t::operator==(const particle_t& other) const {
    return (fabs(x - other.x) < epsilon &&
            fabs(y - other.y) < epsilon &&
            fabs(vx - other.vx) < epsilon &&
            fabs(vy - other.vy) < epsilon &&
            fabs(ax - other.ax) < epsilon &&
            fabs(ay - other.ay) < epsilon);
}

particle_t& particle_t::operator=(const particle_t& other) {
    if (this != &other) {
        id = other.id;
        x = other.x;
        y = other.y;
        vx = other.vx;
        vy = other.vy;
        ax = other.ax;
        ay = other.ay;
    }
    return *this;
}

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


void send_outgoing_parts(int num_particles, particle_t* sending_parts, int send_rank, int destination_rank) {
    // Get the size of the list
    // Send the number of particles first
    MPI_Send(&num_particles, 1, MPI_INT, destination_rank, 0, MPI_COMM_WORLD);

    // Send each particle one by one
    for (int i = 0; i <  num_particles; ++i) {
        particle_t particle = sending_parts[i];
        MPI_Send(&particle, 1, PARTICLE, destination_rank, 0, MPI_COMM_WORLD);
    }
}



void receive_incoming_parts(std::list<particle_t>& incoming_parts, int incoming_rank) {
    MPI_Status status;

    // Receive the number of particles
    int num_particles;
    MPI_Recv(&num_particles, 1, MPI_INT, incoming_rank, 0, MPI_COMM_WORLD, &status);

    // Receive each particle one by one and add to the list
    for (int i = 0; i < num_particles; ++i) {
        particle_t received_particle;
        MPI_Recv(&received_particle, 1, PARTICLE, incoming_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        incoming_parts.push_back(received_particle);
    }
}

// Change particle id to particle index in the array
static int id_to_index(uint64_t id){
    return (int)(id - 1);
}
static int index_to_id(int index){
    return (uint64_t)(index + 1);
}
// Copy particles for communication 
static void part_cpy(particle_t& src_part, particle_t& dst_part){
    dst_part.id = src_part.id;
    dst_part.x = src_part.x;
    dst_part.y = src_part.y;
    dst_part.vx = src_part.vx;
    dst_part.vy = src_part.vy;
    dst_part.ax = src_part.ax;
    dst_part.ay = src_part.ay;
}

std::list<particle_t> incoming_parts;
std::list<int> my_parts;
std::list<int> above_outgoing_parts;
std::list<int> below_outgoing_parts;
std::list<int> ghost_parts;
std::list<int> above_ghost_parts;
std::list<int> below_ghost_parts;
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	double row_width = size / num_procs;
	int total_y = 0;
	int total_ybin = 0;
	std::vector<int> processor_counts(num_procs, 0);
	// Find particles the current processor need to handle (1D layout)
        for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
//	if (rank == 0) {
//		std::cout << "row_width: " << row_width << std::endl;
//		std::cout << "y "<< parts[i].y << std::endl;
//		if (parts[i].y >= 0 && parts[i].y <= size) {
//			++total_y;
//		}
//		int bin = static_cast<int>(parts[i].y / row_width);
//		std::cout << "ybin "<< bin << std::endl;
//		if (bin >= 0 && bin < num_procs) {
//			++total_ybin;
//		}
//	}
        int bin = static_cast<int>(parts[i].y / row_width);
//	std::cout << "bin:" << bin << "without rounding:" << parts[i].y / row_width << std::endl;
        if (bin < 0) {
            std::cout << "bin: " << bin << std::endl;
        }
	if (bin >= num_procs) {
	    std::cout << "bin: " << bin << std::endl;
	}
        if (bin == rank) {
        	my_parts.push_back(i);
		processor_counts[rank]++;
        }
//        if (bin == rank - 1 && abs(parts[i].y - rank * row_width) <= cutoff) {
//    		ghost_parts.push_back(i);
//    	}
//    	if (bin == rank + 1 && abs(parts[i].y - (rank + 1) * row_width) <= cutoff) {
//    		ghost_parts.push_back(i);
//    	}   
	if (rank == 1) { 
	std::cout << "Particle " << i << ": Position = " << parts[i].y << ", Bin = " << bin << ", Rank = " << rank << std::endl;
	}
    }
    std::cout << "Rank: " << rank << std::endl;
    if (rank == 0) {
    	std::cout << "total_y: "<< total_y << std::endl;
	std::cout << "total_ybin: "<< total_ybin << std::endl;
    }
    std::cout << "Init size: " << my_parts.size() << std::endl;
    std::cout << "Particle count for rank " << rank << ": " << processor_counts[rank] << std::endl;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_width = size / num_procs;

	// apply force to all particles in my_parts
    for (int particle : my_parts) {
    	for (int neighbor : my_parts) {
    		apply_force(parts[particle], parts[neighbor]);
    	}
    	for (int neighbor : ghost_parts) {
    		apply_force(parts[particle], parts[neighbor]);
    	}
    }
    ghost_parts.clear();
    std::cout << "Apply force rank: " << rank << std::endl;

    // move particles and find outgoing particles
    for (int particle : my_parts) {
    	move(parts[particle], size);
    	int bin = static_cast<int>(parts[particle].y / row_width);
    	if (bin == rank - 1) {
    		below_outgoing_parts.push_back(particle);
    		my_parts.remove(particle);
    	}
    	if (bin == rank + 1) {
    		above_outgoing_parts.push_back(particle);
    		my_parts.remove(particle);
    	}
    	parts[particle].ax = parts[particle].ay = 0;
    }
    std::cout << "Move particle rank: " << rank << std::endl;

    // redistribute parts, split by even / odd to avoid dead lock
    if (rank % 2 == 0) {
    	if (rank + 1 < num_procs) {
            int num_particles = above_outgoing_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = above_outgoing_parts.begin(); it != above_outgoing_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
    		send_outgoing_parts(num_particles, sending_parts, rank, rank + 1);
            above_outgoing_parts.clear();
            delete[] sending_parts;
    	}
        if (rank - 1 >= 0) {
            int num_particles = below_outgoing_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = below_outgoing_parts.begin(); it != below_outgoing_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
        	send_outgoing_parts(num_particles, sending_parts, rank, rank - 1);
            below_outgoing_parts.clear();
            delete[] sending_parts;
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        
    } else {
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		my_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
            int num_particles = above_outgoing_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = above_outgoing_parts.begin(); it != above_outgoing_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
            send_outgoing_parts(num_particles, sending_parts, rank, rank + 1);
            above_outgoing_parts.clear();
            delete[] sending_parts;
    	}
        if (rank - 1 >= 0) {
            int num_particles = below_outgoing_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = below_outgoing_parts.begin(); it != below_outgoing_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
            send_outgoing_parts(num_particles, sending_parts, rank, rank - 1);
            below_outgoing_parts.clear();
            delete[] sending_parts;
        }
    }
    std::cout << "Send and get rank: " << rank << std::endl;

    // send ghost parts, split by even / odd to avoid dead lock
    for (int particle : my_parts) {
    	if (rank - 1 >= 0 && abs(parts[particle].y - rank * row_width) <= cutoff) {
    		below_ghost_parts.push_back(particle);
    	}
    	if (rank + 1 < num_procs && abs(parts[particle].y - (rank + 1) * row_width) <= cutoff) {
    		above_ghost_parts.push_back(particle);
    	}
    }
    std::cout << "find ghost rank: " << rank << std::endl;
    if (rank % 2 == 0) {
    	if (rank + 1 < num_procs) {
            int num_particles = above_ghost_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = above_ghost_parts.begin(); it != above_ghost_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
            send_outgoing_parts(num_particles, sending_parts, rank, rank + 1);
            above_ghost_parts.clear();
            delete[] sending_parts;
    	}
        if (rank - 1 >= 0) {
            int num_particles = below_ghost_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = below_ghost_parts.begin(); it != below_ghost_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
            send_outgoing_parts(num_particles, sending_parts, rank, rank - 1);
            below_ghost_parts.clear();
            delete[] sending_parts;
        }
        if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
                ghost_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        
    } else {
    	if (rank - 1 >= 0) {
        	receive_incoming_parts(incoming_parts, rank - 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
        	receive_incoming_parts(incoming_parts, rank + 1);
        	for (particle_t& particle : incoming_parts) {
        		ghost_parts.push_back(id_to_index(particle.id));
                part_cpy(particle, parts[id_to_index(particle.id)]);
        	}
        	incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
    		int num_particles = above_ghost_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = above_ghost_parts.begin(); it != above_ghost_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
            send_outgoing_parts(num_particles, sending_parts, rank, rank + 1);
            above_ghost_parts.clear();
            delete[] sending_parts;
    	}
        if (rank - 1 >= 0) {
        	int num_particles = below_ghost_parts.size();
            particle_t* sending_parts = new particle_t[num_particles];
            int index = 0;
            for (std::list<int>::iterator it = below_ghost_parts.begin(); it != below_ghost_parts.end(); ++it) {
                int current_element = *it;
                part_cpy(parts[current_element], sending_parts[index]);
                ++index;
            }
            send_outgoing_parts(num_particles, sending_parts, rank, rank - 1);
            below_ghost_parts.clear();
            delete[] sending_parts;
        }
    }
    std::cout << "Ghost parts rank: " << rank << std::endl;
}

#include <algorithm> // for std::sort

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    
    int* rcounts = nullptr;
    int* displs = nullptr;
    particle_t* rbuf = nullptr;

    // make an array for sending
    particle_t* local_parts_array = new particle_t[my_parts.size()];
    int index = 0;
    for (std::list<int>::iterator it = my_parts.begin(); it != my_parts.end(); ++it) {
        int current_element = *it;
        part_cpy(parts[current_element], local_parts_array[index]);
        ++index;
    }

    // get all my_parts sizes first
    if (rank == 0) {
        rcounts = (int *)malloc(num_procs*sizeof(int)); 
    }
    int num = my_parts.size();
    MPI_Gather( &num, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* root now has correct rcounts, using these we set displs[] so 
     * that data is placed contiguously (or concatenated) at receive end 
     */ 
    if (rank == 0) {
        displs = (int *)malloc(num_procs*sizeof(int)); 
        displs[0] = 0; 
        for (int i = 1; i < num_procs; ++i) { 
            displs[i] = displs[i-1]+rcounts[i-1]; 
        } 
        /* And, create receive buffer  */ 
        rbuf = new particle_t[displs[num_procs-1]+rcounts[num_procs-1]];
    }
    MPI_Gatherv(local_parts_array, my_parts.size(), PARTICLE, rbuf, rcounts, displs, PARTICLE, 0, MPI_COMM_WORLD); 
    delete[] local_parts_array;

    std::cout << "rank: " << rank << std::endl;
    std::cout << "gather size: " << my_parts.size() << std::endl;
    std::cout << "rbuf size: " << displs[num_procs-1]+rcounts[num_procs-1] << std::endl;
    // on root, edit the entire parts array based on info in rbuf
    if (rank == 0) {
        std::cout << "Could gatherv: " << rbuf[0].x << std::endl;
        for (int i = 0; i < displs[num_procs-1]+rcounts[num_procs-1]; ++i) {
            std::cout << "i: " << i << std::endl;
            std::cout << "rbuf: " << rbuf[i].id << std::endl;
            std::cout << "parts: " << parts[id_to_index(rbuf[i].id)].id << std::endl;
            int ori_index = id_to_index(rbuf[i].id);
            part_cpy(rbuf[i], parts[ori_index]);
        }
        std::cout << "Could copy particles: " << rbuf[0].x << std::endl;
        // free memory on root
        free(rcounts);
        free(displs);
        std::cout << "Could free memory: " << rbuf[0].x << std::endl;
    }
    std::cout << "Reach barrier rank: " << rank << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
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
