#include "gl_utils.h"

#include "mesh.h"
#include "mesh_gpu.h"

void GPUMesh::upload()
{
	if (!m)
		return;

	/* Vertex Buffer */
	glGenBuffers(1, &pos_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
	glBufferData(GL_ARRAY_BUFFER,
		     m->positions.size * sizeof(m->positions[0]),
		     m->positions.data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/* Index Buffer */
	glGenBuffers(1, &idx_vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idx_vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		     m->indices.size * sizeof(m->indices[0]), m->indices.data,
		     GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	/* Attribute Buffer */
	if (m->attr.size) {
		glGenBuffers(1, &attr_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, attr_vbo);
		glBufferData(GL_ARRAY_BUFFER, m->attr.size * sizeof(m->attr[0]),
			     m->attr.data, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	/* Vertex Array Object */
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
			      (void *)0);
	glEnableVertexAttribArray(0);
	if (m->attr.size) {
		glBindBuffer(GL_ARRAY_BUFFER, attr_vbo);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
				      1 * sizeof(float), (void *)0);
		glEnableVertexAttribArray(1);
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idx_vbo);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	
	assert(!glGetError());
}

void GPUMesh::draw() const
{
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, m->indices.size, GL_UNSIGNED_INT,
		       (void *)0);
	glBindVertexArray(0);
	
	assert(!glGetError());
}

void GPUMesh::update_attr()
{
	glBindBuffer(GL_ARRAY_BUFFER, attr_vbo);
	glBufferData(GL_ARRAY_BUFFER, m->attr.size * sizeof(m->attr[0]), NULL,
		     GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, m->attr.size * sizeof(m->attr[0]),
		     m->attr.data, GL_STATIC_DRAW);
	
	assert(!glGetError());
}
