cmake_minimum_required(VERSION 3.14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(MU5MAM30)

add_subdirectory(extern/glfw)

###############################################################################
add_executable(test_fem 
	src/bin/test_fem.cpp
	src/common/chrono.cpp
	src/mesh/duplicate_verts.cpp
	src/mesh/cube.cpp
	src/mesh/sphere.cpp
	src/mesh/mesh_io.cpp
	src/fem/fem_matrix.cpp
	src/fem/P1.cpp
	src/linalg/conjugate_gradient.cpp
)
target_include_directories(test_fem PRIVATE 
	include/common
	include/fem
	include/mesh
	include/linalg
	extern
)
target_compile_options(test_fem PUBLIC -Wall -Wextra -ffast-math -march=native)
###############################################################################

###############################################################################
add_executable(test_viewer
	src/bin/test_viewer.cpp
	src/common/chrono.cpp
	src/common/logging.c
	src/mesh/duplicate_verts.cpp
	src/mesh/cube.cpp
	src/mesh/sphere.cpp
	src/mesh/mesh_bounds.cpp
	src/mesh/mesh_gpu.cpp
	src/mesh/mesh_io.cpp
	src/viewer/viewer.cpp
	src/viewer/camera.cpp
	src/viewer/frustum.cpp
	src/viewer/mouse.cpp
	src/viewer/trackball.cpp
	src/viewer/shaders.cpp
	src/fem/fem_matrix.cpp
	src/fem/P1.cpp
	src/linalg/conjugate_gradient.cpp
)
target_include_directories(test_viewer PRIVATE 
	include/common 
	include/mesh 
	include/viewer
	include/fem
	include/linalg
	extern
	extern/glfw/include
)
target_compile_options(test_viewer PUBLIC -Wall -Wextra -Wno-missing-field-initializers)
target_link_libraries(test_viewer PRIVATE glfw imgui_glfw_opengl3 GL tiny_expr)
###############################################################################

###############################################################################
add_executable(test_euler2D
	src/bin/test_euler2D.cpp
	src/common/chrono.cpp
	src/common/logging.c
	src/mesh/duplicate_verts.cpp
	src/mesh/cube.cpp
	src/mesh/sphere.cpp
	src/mesh/mesh_bounds.cpp
	src/mesh/mesh_gpu.cpp
	src/mesh/mesh_io.cpp
	src/viewer/viewer.cpp
	src/viewer/camera.cpp
	src/viewer/frustum.cpp
	src/viewer/mouse.cpp
	src/viewer/trackball.cpp
	src/viewer/shaders.cpp
	src/fem/fem_matrix.cpp
	src/fem/P1.cpp
	src/fem/euler2D.cpp
	src/linalg/conjugate_gradient.cpp
)
target_include_directories(test_euler2D PRIVATE 
	include/common 
	include/mesh 
	include/viewer
	include/fem
	include/linalg
	extern
	extern/glfw/include
)
target_compile_options(test_euler2D PUBLIC -Wall -Wextra -Wno-missing-field-initializers)
target_link_libraries(test_euler2D PRIVATE glfw imgui_glfw_opengl3 GL tiny_expr)
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
target_include_directories(imgui_glfw_opengl3 PRIVATE extern/glfw/include)
###############################################################################

###############################################################################
add_library(tiny_expr
	extern/tiny_expr/tinyexpr.c
)
###############################################################################
