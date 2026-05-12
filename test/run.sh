AMREX_THE_ARENA_INIT_SIZE=2000000000
# amrex.the_arena_init_size = 3000000000

mpirun -np 4 ../build/genesis4 fel.in -o fel_gpu > fel_gpu.log 2>&1

mpirun -np 4 genesis4 fel.in -o fel_cpu > fel_cpu.log 2>&1