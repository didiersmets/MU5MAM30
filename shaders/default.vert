#version 430 core

/* In variables */
layout (location = 0) in vec3 pos;

/* Uniform variables (cell independent) */
layout (location = 0) uniform mat4 vm;
layout (location = 1) uniform mat4 proj;
layout (location = 2) uniform vec3 camera_pos;

/* Out variables */
layout (location = 0) out vec3 V;  /* View vector in world space */
layout (location = 1) out vec3 L;  /* Light vector in world space */

void main() 
{
	V = camera_pos - pos;
	
	/* We assume light comes from camera */
	L = V;
	
	/* Vertex position in clip space */
	gl_Position = proj * vm * vec4(pos, 1.0f);
}
