#version 330 core
in vec3 Normal;

out vec4 color;

uniform vec3 lightDir; 
uniform vec4 in_color;

void main(){
	vec3 normal = normalize(Normal);
	float diffuse_factor = max(dot(normal, normalize(-lightDir)), 0.0);
	color = (0.2 + diffuse_factor)*in_color; 
}