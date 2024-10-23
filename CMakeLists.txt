cmake_minimum_required(VERSION 3.14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(MU5MAM30)

set(GLFW_INSTALL_DIR "./glfw_3.4")
list(APPEND CMAKE_PREFIX_PATH "${GLFW_INSTALL_DIR}")
find_package(glfw3 REQUIRED)

###############################################################################
add_executable(test_fem 
	src/bin/test_fem.cpp
	src/common/chrono.cpp
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
target_compile_options(test_fem PUBLIC -Wall -Wextra)
###############################################################################

###############################################################################
add_executable(test_viewer
	src/bin/test_viewer.cpp
	src/common/chrono.cpp
	src/mesh/cube.cpp
	src/mesh/sphere.cpp
	src/mesh/mesh_bounds.cpp
	src/mesh/mesh_gpu.cpp
	src/mesh/mesh_io.cpp
	src/viewer/gui_app.cpp
	src/viewer/viewer_app.cpp
	src/viewer/camera.cpp
	src/viewer/frustum.cpp
	src/viewer/mouse.cpp
	src/viewer/trackball.cpp
	src/viewer/shaders.cpp
)


target_include_directories(test_viewer PRIVATE 
	include/common 
	include/mesh 
	include/viewer
	include/fem
	include/linalg
	extern
	${GLFW_INCLUDE_DIRS}
)


target_link_libraries(test_viewer PRIVATE glfw imgui_glfw_opengl3 GL)
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


target_include_directories(imgui_glfw_opengl3 PRIVATE ${GLFW_INCLUDE_DIRS})
target_link_libraries(imgui_glfw_opengl3 PRIVATE glfw)
###############################################################################

