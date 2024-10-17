#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <GL/glext.h>

GLuint create_shader(const char *vs_path, const char *fs_path)
{
	int success;
	GLchar infoLog[512];

	GLint vert = glCreateShader(GL_VERTEX_SHADER);
	FILE *f = fopen(vs_path, "rb");
	if (!f) {
		printf("ERROR: Vertex shader %s not found!\n", vs_path);
		return 0;
	}
	fseek(f, 0, SEEK_END);
	int fsize = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *source[] = { NULL };
	source[0] = static_cast<char *>(malloc((fsize + 1)));
	fread(source[0], 1, fsize, f);
	fclose(f);
	source[0][fsize] = '\0';
	glShaderSource(vert, 1, (const char *const *)source, NULL);
	glCompileShader(vert);
	glGetShaderiv(vert, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vert, 512, NULL, infoLog);
		printf("ERROR: compiling of %s shader %s failed !\n%s\n",
		       "GL_VERTEX_SHADER", "vertex", infoLog);
		return 0;
	}

	GLint frag = glCreateShader(GL_FRAGMENT_SHADER);
	FILE *g = fopen(fs_path, "rb");
	if (!g) {
		printf("ERROR: Fragment shader %s not found!\n", fs_path);
		return 0;
	}
	fseek(g, 0, SEEK_END);
	int gsize = ftell(g);
	fseek(g, 0, SEEK_SET);
	char *source2[] = { NULL };
	source2[0] = static_cast<char *>(malloc((gsize + 1)));
	fread(source2[0], 1, gsize, g);
	fclose(g);
	source2[0][gsize] = '\0';
	glShaderSource(frag, 1, (const char *const *)source2, NULL);
	glCompileShader(frag);
	glGetShaderiv(frag, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(frag, 512, NULL, infoLog);
		printf("ERROR: compiling of %s shader %s failed !\n%s\n",
		       "GL_FRAGMENT_SHADER", "frag", infoLog);
		return 0;
	}

	GLint prg = glCreateProgram();
	glAttachShader(prg, vert);
	glAttachShader(prg, frag);
	glLinkProgram(prg);
	glGetProgramiv(prg, GL_LINK_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(frag, 512, NULL, infoLog);
		printf("ERROR: linking failed !\n%s\n", infoLog);
		return 0;
	}
	glDetachShader(prg, vert);
	glDetachShader(prg, frag);
	glDeleteShader(vert);
	glDeleteShader(frag);

	return (prg);
}

