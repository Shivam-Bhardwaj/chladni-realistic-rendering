cmake_minimum_required(VERSION 3.18)
project(ChladniSimulation CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Find audio libraries (Windows with vcpkg)
if(WIN32)
    find_path(PORTAUDIO_INCLUDE_DIR portaudio.h)
    find_library(PORTAUDIO_LIBRARY portaudio)
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    find_library(FFTW3_LIBRARY fftw3f)
else()
    # Linux/Mac with pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
endif()

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 86 89 90)  # RTX 4090 is SM 8.9
enable_language(CUDA)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WIN32)
    include_directories(${PORTAUDIO_INCLUDE_DIR})
    include_directories(${FFTW3_INCLUDE_DIR})
else()
    include_directories(${PORTAUDIO_INCLUDE_DIRS})
    include_directories(${FFTW3_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/main.cpp
)

set(CUDA_SOURCES
    src/particle_physics.cu
)

# Header files (for IDE organization)
set(HEADERS
    include/logger.h
    include/debug_info.h
    include/cuda_memory_manager.h
    include/shader_loader.h
    include/realistic_renderer.h
)

# Create main executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Create demo executable for realistic rendering
add_executable(RealisticRenderingDemo 
    demo_realistic_rendering.cpp 
    ${CUDA_SOURCES}
)

# Set CUDA properties for both executables
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(RealisticRenderingDemo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler flags for both executables
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
)

target_compile_options(RealisticRenderingDemo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
)

# Link libraries for both executables
if(WIN32)
    # Main executable
    target_link_libraries(${PROJECT_NAME}
        CUDA::cudart
        CUDA::curand
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        glm::glm
        imgui::imgui
        ${PORTAUDIO_LIBRARY}
        ${FFTW3_LIBRARY}
    )
    
    # Demo executable (no audio dependencies needed)
    target_link_libraries(RealisticRenderingDemo
        CUDA::cudart
        CUDA::curand
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        glm::glm
    )
else()
    # Main executable
    target_link_libraries(${PROJECT_NAME}
        CUDA::cudart
        CUDA::curand
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        glm::glm
        imgui::imgui
        ${PORTAUDIO_LIBRARIES}
        ${FFTW3_LIBRARIES}
    )
    
    # Demo executable (no audio dependencies needed)
    target_link_libraries(RealisticRenderingDemo
        CUDA::cudart
        CUDA::curand
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        glm::glm
    )
endif()

# Windows-specific settings
if(WIN32)
    # Copy required DLLs to output directory
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:GLEW::GLEW>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()