#include "OBJLoader.h"
#include <iostream>

OBJProperites loadOBJ(std::string file_name) {
	tinyobj::ObjReader reader = tinyobj::ObjReader();
	bool res = reader.ParseFromFile(file_name);

	//VERTICES -----------------------------------------------------------------------

	std::vector<float> obj_vertices = reader.GetAttrib().GetVertices();

	//INDICES ------------------------------------------------------------------------

	std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

	size_t shape_offset = 0;
	std::vector<unsigned int> obj_indices;
	std::vector<unsigned int> normal_indices;
	for (size_t s = 0; s < shapes.size(); s++) {
		size_t face_offset = 0;
		//for each face in mesh
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];
			//for each vertex in face. Number of vertices per face is given by obj file,
			//but if triangulate is specified in reader config then is set to 3 everywhere. 
			for (size_t v = 0; v < fv; v++) {
				//idx is the index in the vertices array
				tinyobj::index_t idx = shapes[s].mesh.indices[face_offset + v];
				obj_indices.push_back((unsigned int)idx.vertex_index);
				//indices[shape_offset + face_offset + v] = idx.vertex_index;
				normal_indices.push_back((unsigned int)idx.normal_index);
			}
			face_offset += fv;
		}
		shape_offset += face_offset;
	}

	//NORMALS ------------------------------------------------------------------------
	/*Embree calculates normals for triangle meshes, so there is no need to create a normal buffer.
	Normals follow right hand rule from vertex order.*/

	std::vector<float> obj_normals = reader.GetAttrib().normals;
	/*OBJs have different indices for vertices, UVs and normals.
	We need to create a new normal vector that stores the normal for vector[i] in the i position*/
	//Note this method is inneficient since it writes the same memory more than one time in some cases.
	float * all_normals = new float[obj_vertices.size()];
	for (int i = 0; i < normal_indices.size(); ++i) {
		unsigned int vertex_index = obj_indices[i];
		unsigned int normal_index = normal_indices[i];

		all_normals[vertex_index * 3] = obj_normals[normal_index * 3];
		all_normals[vertex_index * 3 + 1] = obj_normals[normal_index * 3 + 1];
		all_normals[vertex_index * 3 + 2] = obj_normals[normal_index * 3 + 2];

		//memcpy(&all_normals[vertex_index * 3],
		//	&obj_normals[normal_index * 3],
		//	sizeof(float) * 3);
	}

	return { obj_vertices, obj_indices, all_normals };
}