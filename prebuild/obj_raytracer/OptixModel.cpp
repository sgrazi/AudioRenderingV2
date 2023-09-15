#include "OptixModel.h"
#include "AudioRenderer.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
#include "tinyxml2.h"
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

OptixModel *loadOBJ(const std::string &objFile)
{
    OptixModel *model = new OptixModel;

    const std::string mtlDir = objFile.substr(0, objFile.rfind('/') + 1);
    PRINT(mtlDir);

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
        for (auto faceMatID : shape.mesh.material_ids){
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
                mesh->diffuse = (const vec3f &)materials[materialID].diffuse;
                mesh->diffuse = gdt::randomColor(materialID);
                if (materialID >= 0) {
                    mesh->materialID = materialID;
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

void placeReceiver(Sphere sphere, OptixModel *model, vec3f cameraPosition)
{
    std::set<int> uniqueValues;
    // moving the sphere
    for (const auto& shape : sphere.shapes) {
        std::cout << "HOLAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
        // shape.mesh.indices contains repeated indexes due to the shapes sharing indexes
        // this "for" will make sure that the same index is not overwritten multiple times
        for (const auto& num : shape.mesh.indices) {
            if (uniqueValues.find(num.vertex_index) == uniqueValues.end()) {
                uniqueValues.insert(num.vertex_index);
            }
        }
        for (const auto& index : uniqueValues) {

            // Translate each vertex by (x, y, z)
            std::cout << "POSITION X ORIGINAL " << sphere.original_attributes.vertices[3 * index + 0] << std::endl;
            sphere.attributes.vertices[3 * index + 0] = cameraPosition.x + sphere.original_attributes.vertices[3 * index + 0];
            sphere.attributes.vertices[3 * index + 1] = cameraPosition.y + sphere.original_attributes.vertices[3 * index + 1];
            sphere.attributes.vertices[3 * index + 2] = cameraPosition.z + sphere.original_attributes.vertices[3 * index + 2];
            std::cout << "POSITION X CHANGED " << sphere.attributes.vertices[3 * index + 0] << std::endl;

        }
    }

    std::set<int> materialIDs;
    for (auto faceMatID : sphere.shapes[0].mesh.material_ids)
    {
        materialIDs.insert(faceMatID);
    }

    // converting to optix model
    for (int materialID : materialIDs)
    {
        std::map<tinyobj::index_t, int> knownVertices;
        TriangleMesh* mesh = new TriangleMesh();

        for (int faceID = 0; faceID < sphere.shapes[0].mesh.material_ids.size(); faceID++)
        {
            if (sphere.shapes[0].mesh.material_ids[faceID] != materialID)
                continue;
            tinyobj::index_t idx0 = sphere.shapes[0].mesh.indices[3 * faceID + 0];
            tinyobj::index_t idx1 = sphere.shapes[0].mesh.indices[3 * faceID + 1];
            tinyobj::index_t idx2 = sphere.shapes[0].mesh.indices[3 * faceID + 2];

            vec3i idx(addVertex(mesh, sphere.attributes, idx0, knownVertices),
                      addVertex(mesh, sphere.attributes, idx1, knownVertices),
                      addVertex(mesh, sphere.attributes, idx2, knownVertices));
            mesh->index.push_back(idx);
            // TO DO: check to see if we can remove this, we no longer need visuals
            mesh->diffuse = (const vec3f&)sphere.materials[materialID].diffuse;
            mesh->diffuse = gdt::randomColor(materialID);
            if (materialID >= 0) {
                mesh->materialID = materialID;
            }
        }
        if (mesh->vertex.empty()) {
            delete mesh;
        } else {
            model->meshes.pop_back();
            model->meshes.push_back(mesh);
        }
    }
}