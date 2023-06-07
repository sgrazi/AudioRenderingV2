#version 330 core
out vec4 FragColor;
in vec3 color;

uniform  float time;

void main()
{
	float red = color.x;
	float green = color.y;
	float blue = color.z;

	red = abs(sin(red + time*0.001));
	green = abs(sin(green + time*0.001));
	blue = abs(sin(blue + time*0.001));
	FragColor = vec4(red, green, blue, 1.0f);
};