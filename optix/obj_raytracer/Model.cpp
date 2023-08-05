#include "Model.h"
#include "SampleRenderer.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
//#include "tinyxml2.h"
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

// std::map<int, Material> loadMaterialMap() {
//     std::map<int, Material> materials;

//     // Load and parse the XML file
//     tinyxml2::XMLDocument doc;
//     if (doc.LoadFile("../models/materials.xml") != tinyxml2::XML_SUCCESS) {
//         throw std::runtime_error("Failed to load material XML file");
//     }

//     // Traverse the XML and populate the materials dictionary
//     tinyxml2::XMLElement* materialsNode = doc.FirstChildElement("materials");
//     for (tinyxml2::XMLElement* materialNode = materialsNode->FirstChildElement("material"); materialNode; materialNode = materialNode->NextSiblingElement("material")) {
//         Material material;
//         int id = std::stoi(materialNode->FirstChildElement("id")->GetText());
//         material.name = materialNode->FirstChildElement("name")->GetText();
//         material.ac_absorption = std::stod(materialNode->FirstChildElement("ac_absorption")->GetText());
//         materials[id] = material;
//     }

//     return materials;
// }

Model *loadOBJ(const std::string &objFile)
{
    Model *model = new Model;

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

void placeCamera(Model *model, vec3f cameraPosition)
{ //this does not actually place the sphere in cameraPosition, TO DO
    const std::string objFile = "../models/sphere.obj";
    const std::string mtlDir = objFile.substr(0, objFile.rfind('/') + 1);
    printf("%s",mtlDir.c_str());
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
        throw std::runtime_error("Could not read sphere OBJ model from " + objFile + " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");

    std::set<int> materialIDs;
    for (auto faceMatID : shapes[0].mesh.material_ids)
    {
        materialIDs.insert(faceMatID);
    }

    for (int materialID : materialIDs)
    {
        std::map<tinyobj::index_t, int> knownVertices;
        TriangleMesh* mesh = new TriangleMesh();

        for (int faceID = 0; faceID < shapes[0].mesh.material_ids.size(); faceID++)
        {
            if (shapes[0].mesh.material_ids[faceID] != materialID)
                continue;
            tinyobj::index_t idx0 = shapes[0].mesh.indices[3 * faceID + 0];
            tinyobj::index_t idx1 = shapes[0].mesh.indices[3 * faceID + 1];
            tinyobj::index_t idx2 = shapes[0].mesh.indices[3 * faceID + 2];

            vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                      addVertex(mesh, attributes, idx1, knownVertices),
                      addVertex(mesh, attributes, idx2, knownVertices));
            mesh->index.push_back(idx);
            mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
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