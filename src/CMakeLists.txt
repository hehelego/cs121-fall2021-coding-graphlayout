###
### compiling&linking options
###

# enable openmp
add_compile_options(-fopenmp)
add_link_options(-fopenmp)

# Enable cuda device debugging for Debug build.
# Enable optimization for Release build.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(NOTICE "[build type] debug mode on")
    add_compile_options(
        -O0
        -G
        -g
        -lineinfo
        -Xcompiler=-rdynamic
    )
    add_link_options(
        -g
        -rdynamic
    )
    add_definitions(-D DEBUG)
else()
    message(NOTICE "[build type] debug mode off")
endif()

###
### Libraries
###
find_package(OpenMP)

###
### Executables
###

# Runs on CPU with OpenMP thread parallelism
add_executable(cpu cpu_omp.cu)
# Runs on GPU accelerated by CUDA
add_executable(gpu gpu_cuda.cu)