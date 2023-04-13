#version 330 core

//Check code to see which vbo has each in parameter
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

out vec3 passPosition;
out vec3 Normal;

uniform mat4 worldTransform;
uniform mat4 modelMatrix;

void main(){
	passPosition = position;
	gl_Position = worldTransform * modelMatrix *  vec4(position,1.0);
	Normal = normal;
}