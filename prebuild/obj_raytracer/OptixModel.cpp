#include "OptixModel.h"
#include "AudioRenderer.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
// std
#include <set>
#include <map>

// se usa en xstddef
namespace std
{
    inline bool operator<(const tinyobj::index_t &a,
                          const tinyobj::index_t &b)
    {
        if (a.vertex_index < b.vertex_index)
            return true;
        if (a.vertex_index > b.vertex_index)
            return false;

        if (a.normal_index < b.normal_index)
            return true;
        if (a.normal_index > b.normal_index)
            return false;

        if (a.texcoord_index < b.texcoord_index)
            return true;
        if (a.texcoord_index > b.texcoord_index)
            return false;

        return false;
    }
}

/*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
int addVertex(TriangleMesh *mesh,
              tinyobj::attrib_t &attributes,
              const tinyobj::index_t &idx,
              std::map<tinyobj::index_t, int> &knownVertices)
{
    if (knownVertices.find(idx) != knownVertices.end())
        return knownVertices[idx];

    const vec3f *vertex_array = (const vec3f *)attributes.vertices.data();
    const vec3f *normal_array = (const vec3f *)attributes.normals.data();
    const vec2f *texcoord_array = (const vec2f *)attributes.texcoords.data();

    int newID = mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0)
    {
        while (mesh->normal.size() < mesh->vertex.size())
            mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0)
    {
        while (mesh->texcoord.size() < mesh->vertex.size())
            mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
        mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
        mesh->normal.resize(mesh->vertex.size());

    return newID;
}

// OptixModel *loadOBJ(const std::string &objFile, tinyxml2::XMLDocument &xml_dic)
OptixModel *loadOBJ(const std::string &objFile)
{
    OptixModel *model = new OptixModel;

    const std::string mtlDir = objFile.substr(0, objFile.rfind('/') + 1);

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK = tinyobj::LoadObj(&attributes,
                                   &shapes,
                                   &materials,
                                   &err,
                                   &err,
                                   objFile.c_str(),
                                   mtlDir.c_str(),
                                   /* triangulate */ true);
    if (!readOK)
    {
        throw std::runtime_error("Could not read OBJ model from " + objFile + ":" + mtlDir + " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++)
    {
        tinyobj::shape_t &shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
        {
            materialIDs.insert(faceMatID);
        }

        for (int materialID : materialIDs)
        {
            std::map<tinyobj::index_t, int> knownVertices;
            TriangleMesh *mesh = new TriangleMesh();

            for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++)
            {
                if (shape.mesh.material_ids[faceID] != materialID)
                    continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                          addVertex(mesh, attributes, idx1, knownVertices),
                          addVertex(mesh, attributes, idx2, knownVertices));
                mesh->index.push_back(idx);
                if (materialID >= 0)
                {
                    mesh->material_name = materials[materialID].name;
                }
            }

            if (mesh->vertex.empty())
                delete mesh;
            else
                model->meshes.push_back(mesh);
        }
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
        for (auto vtx : mesh->vertex)
            model->bounds.extend(vtx);

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
}

void placeReceiver(Sphere sphere, OptixModel *model, vec3f cameraPosition, float rotation)
{
    place_receiver_half(sphere.get_left_side(), model, cameraPosition, true, rotation);
    place_receiver_half(sphere.get_right_side(), model, cameraPosition, false, rotation);
}

void place_receiver_half(HalfSphere side, OptixModel *model, vec3f cameraPosition, bool is_left, float rotation)
{
    // moving the side
    for (const auto &shape : side.shapes)
    {
        std::set<int> uniqueValues;
        // shape.mesh.indices contains repeated indexes due to the shapes sharing indexes
        // this "for" will make sure that the same index is not overwritten multiple times
        for (const auto &num : shape.mesh.indices)
        {
            if (uniqueValues.find(num.vertex_index) == uniqueValues.end())
            {
                uniqueValues.insert(num.vertex_index);
            }
        }
        for (const auto &index : uniqueValues)
        {

            // Asumiendo que rotation est� en grados, convi�rtelo a radianes
            float angleRadians = glm::radians(rotation);

            // Crea una matriz de transformaci�n que incluya la rotaci�n
            glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), -angleRadians, glm::vec3(0, 1, 0));

            // Aplicar la rotaci�n
            glm::vec4 vert = glm::vec4(side.original_attributes.vertices[3 * index + 0],
                                       side.original_attributes.vertices[3 * index + 1],
                                       side.original_attributes.vertices[3 * index + 2],
                                       1.0);
            vert = rotationMatrix * vert;

            // Translate each vertex by (x, y, z)
            side.attributes.vertices[3 * index + 0] = cameraPosition.x + vert.x;
            side.attributes.vertices[3 * index + 1] = cameraPosition.y + vert.y;
            side.attributes.vertices[3 * index + 2] = cameraPosition.z + vert.z;
        }
    }

    std::set<int> materialIDs;
    for (auto faceMatID : side.shapes[0].mesh.material_ids)
    {
        materialIDs.insert(faceMatID);
    }

    // converting to optix model
    for (int materialID : materialIDs)
    {
        std::map<tinyobj::index_t, int> knownVertices;
        TriangleMesh *mesh = new TriangleMesh();

        for (int faceID = 0; faceID < side.shapes[0].mesh.material_ids.size(); faceID++)
        {
            if (side.shapes[0].mesh.material_ids[faceID] != materialID)
                continue;
            tinyobj::index_t idx0 = side.shapes[0].mesh.indices[3 * faceID + 0];
            tinyobj::index_t idx1 = side.shapes[0].mesh.indices[3 * faceID + 1];
            tinyobj::index_t idx2 = side.shapes[0].mesh.indices[3 * faceID + 2];

            vec3i idx(addVertex(mesh, side.attributes, idx0, knownVertices),
                      addVertex(mesh, side.attributes, idx1, knownVertices),
                      addVertex(mesh, side.attributes, idx2, knownVertices));
            mesh->index.push_back(idx);
            if (is_left)
            {
                mesh->material_name = "receiver_left";
                mesh->material_absorption = -1;
            }
            else
            {
                mesh->material_name = "receiver_right";
                mesh->material_absorption = -2;
            }
        }
        if (mesh->vertex.empty())
        {
            delete mesh;
        }
        else
        {
            if (is_left)
            {
                // Delete left sphere
                model->meshes.erase(std::remove_if(model->meshes.begin(), model->meshes.end(), [](TriangleMesh *aux_mesh)
                                                   { return aux_mesh->material_absorption == -1; }),
                                    model
                                        ->meshes.end());
            }
            else
            {
                // Delete right sphere
                model->meshes.erase(std::remove_if(model->meshes.begin(), model->meshes.end(), [](TriangleMesh *aux_mesh)
                                                   { return aux_mesh->material_absorption == -2; }),
                                    model
                                        ->meshes.end());
            }
            model->meshes.push_back(mesh);
        }
    }
}
