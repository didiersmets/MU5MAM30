#version 430 core

// In variables
layout (location = 0) in vec3 _V;  /* View vector  (non normalized) */
layout (location = 1) in vec3 _L;   /* Light vector (non normalized) */

/* Uniform variables (cell independent) */
layout (location = 3) uniform bool lighting;

// Out color
layout (location = 0) out vec4 color;

#define EDGE_COLOR vec3(0.f, 0.1f, 0.f) 
#define AMBIENT_COLOR vec3(1.f, 0.8f, 1.f)
#define Ka 0.1f
#define DIFFUSE_COLOR vec3(.88f, .75f, 0.43f)
#define Kd .8f
#define SPECULAR_COLOR vec3(1.f, 1.f, 1.f)
#define Ks 0.1f
#define shininess 8

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
		vec3 full = Ka * AMBIENT_COLOR
		          + Kd * Id * DIFFUSE_COLOR
		          + Ks * Is * SPECULAR_COLOR;
		color = vec4(full, 1.0f);
	}
}
