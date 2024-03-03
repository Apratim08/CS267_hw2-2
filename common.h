#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdint>
#include <mpi.h>
#include <cmath>

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

typedef struct particle_t {
    uint64_t id; // Particle ID
    double x;    // Position X
    double y;    // Position Y
    double vx;   // Velocity X
    double vy;   // Velocity Y
    double ax;   // Acceleration X
    double ay;   // Acceleration Y
    //void* data;

    // Define the equality operator for particle_t
    const double epsilon = 1e-9; // Adjust epsilon based on your application's requirements

    bool operator==(const particle_t& other) const;
    particle_t& operator=(const particle_t& other);

} particle_t;

extern MPI_Datatype PARTICLE;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs);
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs);
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs);

#endif