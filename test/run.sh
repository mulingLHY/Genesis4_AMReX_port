AMREX_THE_ARENA_INIT_SIZE=0
# amrex.the_arena_init_size = 0

mpirun -np 4 ../build/genesis4 fel.in -o fel_gpu > fel_gpu.log 2>&1

# change &track use_cuda=0 to run with cpu
mpirun -np 4 ../build/genesis4 fel.in -o fel_cpu > fel_cpu.log 2>&1