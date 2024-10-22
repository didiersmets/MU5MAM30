#version 430 core

/* In variables */
layout (location = 0) in vec3 pos_;
layout (location = 1) in float attr_;

/* Uniform variables (cell independent) */
layout (location = 0) uniform mat4 vm;
layout (location = 1) uniform mat4 proj;
layout (location = 2) uniform vec3 camera_pos;
layout (location = 4) uniform float scale_min;
layout (location = 5) uniform float scale_max;

/* Out variables */
layout (location = 0) out vec3 V;    /* View vector in world space  */
layout (location = 1) out vec3 L;    /* Light vector in world space */
layout (location = 2) out float u;   /* Value of the solution */

void main() 
{
	u = attr_;
	vec3 pos = pos_ * (1.f + .0f * (u - scale_min) / (scale_max - scale_min));
	
	V = camera_pos - pos;
	
	/* We assume light comes from camera */
	L = V;
	
	/* Vertex position in clip space */
	gl_Position = proj * vm * vec4(pos, 1.0f);

}
