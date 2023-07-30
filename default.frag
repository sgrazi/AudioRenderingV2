#version 330 core
out vec4 FragColor;

in vec3 pos;
in vec3 Normal;
in vec3 crntPos;
in vec3 color;
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

	FragColor = vec4(color, 1.0f) * lightColor * diffuse;
};