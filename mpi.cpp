#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <list>
#include <cstdlib> // for exit()


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
    for (auto it = outgoing_parts->begin(); it != outgoing_parts->end(); ++it) {
        particle_t& particle = *it;
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
std::vector<std::list<particle_t>> my_parts;
std::list<particle_t> above_outgoing_parts;
std::list<particle_t> below_outgoing_parts;
std::vector<std::list<particle_t>> ghost_parts;
std::list<particle_t> above_ghost_parts;
std::list<particle_t> below_ghost_parts;
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_width = size / num_procs;
    int bucket_num = static_cast<int>(size / cutoff);
    double col_width = size / bucket_num;
    my_parts.resize(bucket_num);
    ghost_parts.resize(bucket_num);

    std::vector<std::list<particle_t>> otherghost;
    otherghost.resize(num_procs);
    // Find particles the current processor need to handle (1D layout)
    if (rank == 0) {
        std::vector<std::list<particle_t>> outsending;
        outsending.resize(num_procs);
        for (int i = 0; i < num_parts; ++i) {
            parts[i].ax = parts[i].ay = 0;
            int bin = static_cast<int>(parts[i].y / row_width);
            // get particles to other processors
            outsending[bin].push_back(parts[i]);
            // get ghost parts to other processors
            if (bin - 1 >= 0 && abs(parts[i].y - bin * row_width) <= cutoff) {
                otherghost[bin - 1].push_back(parts[i]);
            }
            if (bin + 1 < num_procs && abs(parts[i].y - (bin + 1) * row_width) <= cutoff) {
                otherghost[bin + 1].push_back(parts[i]);
            }
            if (bin == rank){
                int xindex = static_cast<int>(parts[i].x / col_width);
                my_parts[xindex].push_back(parts[i]);
            }
            if (bin == rank - 1 && abs(parts[i].y - rank * row_width) <= cutoff) {
                int xindex = static_cast<int>(parts[i].x / col_width);
                ghost_parts[xindex].push_back(parts[i]);
            }
            if (bin == rank + 1 && abs(parts[i].y - (rank + 1) * row_width) <= cutoff) {
                int xindex = static_cast<int>(parts[i].x / col_width);
                ghost_parts[xindex].push_back(parts[i]);
            }    
        }
        for (int out_proc = 1; out_proc < num_procs; ++out_proc) {
            std::list<particle_t> parts_out = outsending[out_proc];
            send_outgoing_parts(&parts_out, 0, out_proc);
            parts_out.clear();
        }
    } else {
        receive_incoming_parts(incoming_parts, 0);
        for (particle_t& particle : incoming_parts) {
            int xindex = static_cast<int>(particle.x / col_width);
            my_parts[xindex].push_back(particle);
        }
        incoming_parts.clear();
    }
    if (rank == 0) {
        for (int out_proc = 1; out_proc < num_procs; ++out_proc) {
            std::list<particle_t> parts_out = otherghost[out_proc];
            send_outgoing_parts(&parts_out, 0, out_proc);
            parts_out.clear();
        }
    } else {
        receive_incoming_parts(incoming_parts, 0);
        for (particle_t& particle : incoming_parts) {
            int xindex = static_cast<int>(particle.x / col_width);
            ghost_parts[xindex].push_back(particle);
        }
        incoming_parts.clear();
    }
}
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_width = size / num_procs;
    int bucket_num = static_cast<int>(size / cutoff);
    double col_width = size / bucket_num;

    // apply force to all particles in my_parts
    for (int i = 0; i < bucket_num; ++i) {
        for (particle_t& particle : my_parts[i]) {
            for (int j = -1; j < 2; ++j) {
                if (i + j >= 0 && i + j < bucket_num) {
                    for (particle_t& neighbor : my_parts[i + j]) {
                        apply_force(particle, neighbor);
                    }
                    for (particle_t& neighbor : ghost_parts[i + j]) {
                        apply_force(particle, neighbor);
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < bucket_num; ++i) {
        ghost_parts[i].clear();
    }
    
    // move particles and find outgoing particles
    std::vector<std::list<particle_t>> move_bins;
    move_bins.resize(bucket_num);
    for (int i = 0; i < bucket_num; ++i) {
        for (auto it = my_parts[i].begin(); it != my_parts[i].end();) {
            particle_t& particle = *it;
            move(particle, size);
            particle.ax = particle.ay = 0;
            int bin = static_cast<int>(particle.y / row_width);
            if (bin == rank - 1) {
                below_outgoing_parts.push_back(particle);
                it = my_parts[i].erase(it);  // Erase the element and advance the iterator
            }
            else if (bin == rank + 1) {
                above_outgoing_parts.push_back(particle);
                it = my_parts[i].erase(it);  // Erase the element and advance the iterator
            }
            else {
                int xindex = static_cast<int>(particle.x / col_width);
                if (xindex != i) {
                    move_bins[xindex].push_back(particle);
                    it = my_parts[i].erase(it);  // Erase the element and advance the iterator
                } else {
                    ++it;  // Move to the next element
                }
            }
        }
    }
    for (int i = 0; i < bucket_num; ++i) {
        for (particle_t& particle : move_bins[i]) {
            my_parts[i].push_back(particle);
        }
        move_bins[i].clear();
    }

    // redistribute parts, split by even / odd to avoid dead lock
    if (rank % 2 == 0) {
        if (rank + 1 < num_procs) {
            send_outgoing_parts(&above_outgoing_parts, rank, rank + 1);
            above_outgoing_parts.clear();
        }
        if (rank - 1 >= 0) {
            send_outgoing_parts(&below_outgoing_parts, rank, rank - 1);
            below_outgoing_parts.clear();
        }
        if (rank + 1 < num_procs) {
            receive_incoming_parts(incoming_parts, rank + 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                my_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                my_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        
    } else {
        if (rank + 1 < num_procs) {
            receive_incoming_parts(incoming_parts, rank + 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                my_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                my_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
            send_outgoing_parts(&above_outgoing_parts, rank, rank + 1);
            above_outgoing_parts.clear();
        }
        if (rank - 1 >= 0) {
            send_outgoing_parts(&below_outgoing_parts, rank, rank - 1);
            below_outgoing_parts.clear();
        }
    }

    // send ghost parts, split by even / odd to avoid dead lock
    for (int i = 0; i < bucket_num; ++i) {
        // Iterate over each particle in the list
        for (particle_t& particle : my_parts[i]) {
            if (rank - 1 >= 0 && abs(particle.y - rank * row_width) <= cutoff) {
                below_ghost_parts.push_back(particle);
            }
            if (rank + 1 < num_procs && abs(particle.y - (rank + 1) * row_width) <= cutoff) {
                above_ghost_parts.push_back(particle);
            }
        }
    }
    if (rank % 2 == 0) {
        if (rank + 1 < num_procs) {
            send_outgoing_parts(&above_ghost_parts, rank, rank + 1);
            above_ghost_parts.clear();
        }
        if (rank - 1 >= 0) {
            send_outgoing_parts(&below_ghost_parts, rank, rank - 1);
            below_ghost_parts.clear();
        }
        if (rank - 1 >= 0) {
            incoming_parts.clear();
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                ghost_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
            incoming_parts.clear();
            receive_incoming_parts(incoming_parts, rank + 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                ghost_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        
    } else {
        if (rank - 1 >= 0) {
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                ghost_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
            receive_incoming_parts(incoming_parts, rank + 1);
            for (particle_t& particle : incoming_parts) {
                int xindex = static_cast<int>(particle.x / col_width);
                ghost_parts[xindex].push_back(particle);
            }
            incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
            send_outgoing_parts(&above_ghost_parts, rank, rank + 1);
            above_ghost_parts.clear();
        }
        if (rank - 1 >= 0) {
            send_outgoing_parts(&below_ghost_parts, rank, rank - 1);
            below_ghost_parts.clear();
        }
    }
}
#include <algorithm> // for std::sort
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_width = size / num_procs;
    int bucket_num = static_cast<int>(size / cutoff);
    double col_width = size / bucket_num;

    std::list<particle_t> rbuf;
    // make an array for sending
    std::list<particle_t> local_parts_array;
    for (int i = 0; i < bucket_num; ++i) {
        // Iterate over each particle in the list
        for (particle_t& particle : my_parts[i]) {
            // Copy the particle to the current position in local_parts_array
            local_parts_array.push_back(particle);
        }
    }

    if (rank == 0) {
        for (std::list<particle_t>& particle_list : my_parts) {
            // Iterate over each particle in the list
            for (particle_t& particle : particle_list) {
                if (particle.id <= 0 || particle.id > num_parts) {
                    std::cout << "rank: " << 0 << " has weired paritcle:" << particle.id << std::endl;
                }
                part_cpy(particle, parts[id_to_index(particle.id)]);
            }
        }
        for (int i = 1; i < num_procs; ++i) {
            receive_incoming_parts(rbuf, i);
            for (particle_t& particle : rbuf) {
                if (particle.id <= 0 || particle.id > num_parts) {
                    std::cout << "rank: " << i << " send weired paritcle:" << particle.id << std::endl;
                }
                part_cpy(particle, parts[id_to_index(particle.id)]);
            }
            rbuf.clear();
        }
    } else {
        send_outgoing_parts(&local_parts_array, rank, 0);
    }
}
