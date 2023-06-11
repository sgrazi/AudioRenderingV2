#version 330 core
out vec4 FragColor;
in vec3 pos;

uniform  float time;

void main()
{
	float x = pos.x;
	float y = pos.y;
	float z = pos.z;

	float red = abs(sin(0.34 + y + time*0.001));
	float green = abs(cos(0.74 + y + time*0.001));
	float blue = abs(tan(0.17 + y + time*0.001));
	FragColor = vec4(red, green, blue, 1.0f);
};