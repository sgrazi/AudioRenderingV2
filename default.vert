#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

// Inputs the matrices needed for 3D view
uniform mat4 camMatrix;

out vec3 color;
void main()
{
   gl_Position = camMatrix * vec4(aPos, 1.0);
   color = aColor;
};