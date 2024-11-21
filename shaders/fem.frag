#version 410 core

// In variables
layout (location = 0) in vec3 _V;  /* View vector  (non normalized) */
layout (location = 1) in vec3 _L;   /* Light vector (non normalized) */
layout (location = 2) in float u;

/* Uniform variables (cell independent) */
uniform mat4 vm;
uniform mat4 proj;
uniform vec3 camera_pos;
uniform bool lighting;
uniform float scale_min;
uniform float scale_max;
uniform float deform;

// Out color
layout (location = 0) out vec4 color;

#define EDGE_COLOR vec3(0.f, 0.1f, 0.f) 
#define Ka 0.1f
#define Kd .8f
#define SPECULAR_COLOR vec3(1.f, 1.f, 1.f)
#define Ks 0.1f
#define shininess 8

vec3 color_from_val(float u, float scale_min, float scale_max)
{
	vec3 color;
	float l = (u - scale_min) / (scale_max - scale_min);
	if (l < 0.5f) {
		color = vec3(1 - 2.f * l, 2.f * l, 0.f);
	} else {
		l = 1. - l;
		color = vec3(0.f, 2.f * l, 1 - 2.f * l);
	}
	return color;
}


void main() 
{
	if (!lighting)
	{
		color = vec4(EDGE_COLOR, 1.0f);
	} else /* flat shading */ {
		/**
		 * Yields face normal in world coords.
		 * Using non normalized _V is equivalent to using world pos but avoids an additional 'in' variable
		 */
		vec3 N = normalize(cross(dFdx(_V), dFdy(_V)));
		vec3 V = normalize(_V);
                vec3 L = normalize(_L);
		float Id = max(dot(N, L), 0.f);
		float Is = 0.f;
		if (Id > 0) {
			vec3 R = reflect(-L, N);
			float ca = max(dot(R, V), 0.f); 
			Is = pow(ca, shininess) * shininess / 4;
		}
		vec3 base = color_from_val(u, scale_min, scale_max);
		vec3 full = Ka * base
		          + Kd * Id * base
		          + Ks * Is * SPECULAR_COLOR;
		color = vec4(full, 1.0f);
	}
}
