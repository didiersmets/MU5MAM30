#version 410 core

/* In variables */
layout (location = 0) in vec3 pos_;
layout (location = 1) in float attr_;

/* Uniform variables (cell independent) */
uniform mat4 vm;
uniform mat4 proj;
uniform vec3 camera_pos;
uniform bool lighting;
uniform float scale_min;
uniform float scale_max;
uniform float deform;

/* Out variables */
layout (location = 0) out vec3 _V;    /* View vector in world space  */
layout (location = 1) out vec3 _L;    /* Light vector in world space */
layout (location = 2) out float u;   /* Value of the solution */

void main() 
{
	u = attr_;
	vec3 pos = pos_ * (1.f + deform * (u - scale_min) / (scale_max - scale_min));
	
	_V = camera_pos - pos;
	
	/* We assume light comes from camera */
	_L = _V;
	
	/* Vertex position in clip space */
	gl_Position = proj * vm * vec4(pos, 1.0f);

}
