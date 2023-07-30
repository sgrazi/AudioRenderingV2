#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

// Inputs the matrices needed for 3D view
uniform mat4 camMatrix;

uniform mat4 model;

out vec3 pos;
out vec3 Normal;
out vec3 crntPos;
out vec3 color;

void main()
{
	crntPos = vec3(model * vec4(aPos, 1.0f));
	
	gl_Position = camMatrix * vec4(crntPos, 1.0);
	pos = aPos;
	Normal = aNormal;
	color = aColor;
};