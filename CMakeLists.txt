cmake_minimum_required(VERSION 3.14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

project(MU5MAM30 LANGUAGES CXX C)

find_package(OpenGL REQUIRED)

###############################################################################
set(USE_OPENMP FALSE)
if (USE_OPENMP)
	find_package(OpenMP)
endif()
add_library(nanofem
	src/mesh/adjacency.cpp
	src/mesh/cube.cpp
	src/mesh/sphere.cpp
	src/mesh/duplicate_verts.cpp
	src/mesh/mesh_io.cpp
	src/mesh/mesh_bounds.cpp
	src/linalg/sparse_matrix.cpp
	src/linalg/conjugate_gradient.cpp
	src/fem/fem_matrix.cpp
	src/fem/P1.cpp
	src/fem/poisson.cpp
	src/fem/navier_stokes.cpp
)
target_include_directories(nanofem PUBLIC 
	include/common 
	include/mesh 
	include/fem
	include/linalg
	extern
)
target_compile_options(nanofem PRIVATE 
	-std=c++17 -Wall -Wextra -ffast-math -march=native -Wno-unknown-pragmas
)
if (USE_OPENMP AND OpenMP_CXX_FOUND)
	target_compile_options(nanofem PRIVATE -DUSE_OPENMP)
	target_link_libraries(nanofem PUBLIC OpenMP::OpenMP_CXX)
endif()
###############################################################################

###############################################################################
add_library(nanoviewer
	src/mesh/mesh_gpu.cpp
	src/viewer/viewer.cpp
	src/viewer/camera.cpp
	src/viewer/frustum.cpp
	src/viewer/mouse.cpp
	src/viewer/trackball.cpp
	src/viewer/shaders.cpp
)
target_include_directories(nanoviewer PUBLIC 
	include/common 
	include/mesh
	include/viewer
	extern
)
target_compile_options(nanoviewer PRIVATE
	-std=c++17 -Wall -Wextra
)
target_link_libraries(nanoviewer PUBLIC
	glfw
	imgui_glfw_opengl3
	OpenGL::GL
)
###############################################################################

###############################################################################
add_executable(test_poisson
	src/common/logging.cpp
	src/bin/test_poisson.cpp
)
target_compile_options(test_poisson PRIVATE 
	-std=c++17 -Wall -Wextra -Wno-missing-field-initializers
)
target_link_libraries(test_poisson PRIVATE nanofem nanoviewer tiny_expr) 
###############################################################################

###############################################################################
add_executable(test_NS
	src/common/logging.cpp
	src/bin/test_navier_stokes.cpp
)
target_compile_options(test_NS PRIVATE 
	-std=c++17 -Wall -Wextra -Wno-missing-field-initializers
)
target_link_libraries(test_NS PRIVATE nanofem nanoviewer tiny_expr)
###############################################################################

###############################################################################
add_library(imgui_glfw_opengl3
	extern/imgui/imgui.cpp
	extern/imgui/imgui_draw.cpp
	extern/imgui/imgui_tables.cpp
	extern/imgui/imgui_widgets.cpp
	extern/imgui/imgui_impl_glfw.cpp
	extern/imgui/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_glfw_opengl3 PUBLIC
	extern/glfw/include
)
###############################################################################

###############################################################################
add_library(tiny_expr
	extern/tiny_expr/tinyexpr.c
)
target_include_directories(tiny_expr PUBLIC
	extern
)
###############################################################################

add_subdirectory(extern/glfw)

