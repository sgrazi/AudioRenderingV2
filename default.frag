#version 330 core
out vec4 FragColor;

in vec3 pos;

in vec3 Normal;
in vec3 crntPos;

uniform  float time;

uniform vec4 lightColor;

uniform vec3 lightPos;

void main()
{
	vec3 normal = normalize(Normal);
	vec3 lightDirection = normalize(lightPos - crntPos);

	float diffuse = max(dot(normal, lightDirection), 0.0f);

	float x = pos.x;
	float y = pos.y;
	float z = pos.z;

	float red = 0.23f;
	float green = 0.7f;
	float blue = 0.22f;

	FragColor = vec4(red, green, blue, 1.0f) * lightColor * diffuse;
};