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
        particle_t received_particle;
        MPI_Recv(&received_particle, sizeof(particle_t), MPI_BYTE, incoming_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
std::list<particle_t> my_parts;
std::list<particle_t> above_outgoing_parts;
std::list<particle_t> below_outgoing_parts;
std::list<particle_t> ghost_parts;
std::list<particle_t> above_ghost_parts;
std::list<particle_t> below_ghost_parts;
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_width = size / num_procs;
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
                my_parts.push_back(parts[i]);
            }
            if (bin == rank - 1 && abs(parts[i].y - rank * row_width) <= cutoff) {
                ghost_parts.push_back(parts[i]);
            }
            if (bin == rank + 1 && abs(parts[i].y - (rank + 1) * row_width) <= cutoff) {
                ghost_parts.push_back(parts[i]);
            }    
        }
        for (int out_proc = 1; out_proc < num_procs; ++out_proc) {
            std::list<particle_t> parts_out = outsending[out_proc];
            send_outgoing_parts(&parts_out, 0, out_proc);
        }
    } else {
        receive_incoming_parts(my_parts, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int out_proc = 1; out_proc < num_procs; ++out_proc) {
            std::list<particle_t> parts_out = otherghost[out_proc];
            send_outgoing_parts(&parts_out, 0, out_proc);
        }
    } else {
        receive_incoming_parts(ghost_parts, 0);
    }
    std::cout << "Rank: " << rank << std::endl;
    std::cout << "Init size: " << my_parts.size() << std::endl;
    std::cout << "Current ghost_parts:" << ghost_parts.size() << std::endl;
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
    std::cout << "Could apply force rank: " << rank << std::endl;
    ghost_parts.clear();
    // move particles and find outgoing particles
    for (particle_t& particle : my_parts) {
        move(particle, size);
        particle.ax = particle.ay = 0;
        int bin = static_cast<int>(particle.y / row_width);
        if (bin == rank - 1) {
            std::cout << "start push below: " << below_outgoing_parts.size() << std::endl;
            particle_t copy_part;
            part_cpy(particle, copy_part);
            below_outgoing_parts.push_back(copy_part);
            my_parts.remove(particle);
            std::cout << "end push below: " << below_outgoing_parts.size() << std::endl;
        }
        if (bin == rank + 1) {
            std::cout << "start push above: " << above_outgoing_parts.size() << std::endl;
            particle_t copy_part;
            part_cpy(particle, copy_part);
            above_outgoing_parts.push_back(copy_part);
            my_parts.remove(particle);
            std::cout << "end push above: " << above_outgoing_parts.size() << std::endl;
        }
    }
    std::cout << "Could move rank: " << rank << std::endl;
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
                particle_t copy_part;
                part_cpy(particle, copy_part);
                my_parts.push_back(copy_part);
            }
            incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                particle_t copy_part;
                part_cpy(particle, copy_part);
                my_parts.push_back(copy_part);
            }
            incoming_parts.clear();
        }
        
    } else {
        if (rank + 1 < num_procs) {
            receive_incoming_parts(incoming_parts, rank + 1);
            for (particle_t& particle : incoming_parts) {
                particle_t copy_part;
                part_cpy(particle, copy_part);
                my_parts.push_back(copy_part);
            }
            incoming_parts.clear();
        }
        if (rank - 1 >= 0) {
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                particle_t copy_part;
                part_cpy(particle, copy_part);
                my_parts.push_back(copy_part);
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
    std::cout << "send and get rank: " << rank << std::endl;
    // send ghost parts, split by even / odd to avoid dead lock
    for (particle_t& particle : my_parts) {
        if (rank - 1 >= 0 && abs(particle.y - rank * row_width) <= cutoff) {
            particle_t copy_part;
            part_cpy(particle, copy_part);
            below_ghost_parts.push_back(copy_part);
        }
        if (rank + 1 < num_procs && abs(particle.y - (rank + 1) * row_width) <= cutoff) {
            particle_t copy_part;
            part_cpy(particle, copy_part);
            above_ghost_parts.push_back(copy_part);
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
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                particle_t copy_part;
                part_cpy(particle, copy_part);
                ghost_parts.push_back(copy_part);
            }
            incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
            receive_incoming_parts(incoming_parts, rank + 1);
            for (particle_t& particle : incoming_parts) {
                particle_t copy_part;
                part_cpy(particle, copy_part);
                ghost_parts.push_back(copy_part);
            }
            incoming_parts.clear();
        }
        
    } else {
        if (rank - 1 >= 0) {
            receive_incoming_parts(incoming_parts, rank - 1);
            for (particle_t& particle : incoming_parts) {
                particle_t copy_part;
                part_cpy(particle, copy_part);
                ghost_parts.push_back(copy_part);
            }
            incoming_parts.clear();
        }
        if (rank + 1 < num_procs) {
            receive_incoming_parts(incoming_parts, rank + 1);
            for (particle_t& particle : incoming_parts) {
                particle_t copy_part;
                part_cpy(particle, copy_part);
                ghost_parts.push_back(copy_part);
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
    std::cout << "get ghost parts rank: " << rank << std::endl;
    std::cout << "rank: " << rank << " current my_parts:" << my_parts.size() << " current ghost_parts:" << ghost_parts.size() << std::endl;
}
#include <algorithm> // for std::sort
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    
    int* rcounts = nullptr;
    int* displs = nullptr;
    particle_t* rbuf = nullptr;
    // make an array for sending
    particle_t* local_parts_array = new particle_t[my_parts.size()];
    std::copy(my_parts.begin(), my_parts.end(), local_parts_array);
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
        rbuf = new particle_t[num_parts];
    }
    MPI_Gatherv(local_parts_array, my_parts.size(), PARTICLE, rbuf, rcounts, displs, PARTICLE, 0, MPI_COMM_WORLD); 
    delete[] local_parts_array;

    // on root, edit the entire parts array based on info in rbuf
    if (rank == 0) {
        std::cout << "Could gatherv: " << rbuf[0].x << std::endl;
        for (int i = 0; i < num_parts; ++i) {
            int ori_index = id_to_index(rbuf[i].id);
            part_cpy(rbuf[i], parts[ori_index]);
        }
        std::cout << "Could copy particles: " << rbuf[0].x << std::endl;
        // free memory on root
        free(rcounts);
        free(displs);
        delete[] rbuf;
        std::cout << "Could free memory: " << rbuf[0].x << std::endl;
    }
    std::cout << "Rank: " << rank << " my_parts: " << my_parts.size() << std::endl;
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
