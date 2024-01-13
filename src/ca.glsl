#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

layout(set = 0, binding = 1) readonly buffer LifeInBuffer {
    uint life_in[];
};

layout(set = 0, binding = 2) writeonly buffer LifeOutBuffer {
    uint life_out[];
};

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);


    uint x = life_in[pos.x * 16 + pos.y];

    vec3 cvec;

    if (x == 0) {
        cvec = vec3(0.0863, 0.2392, 0.6588);
    }
    else {
        cvec = vec3(0.6275, 0.2314, 0.2314);
    }

    life_out[0] = x; /// Do rules. 
    imageStore(img, pos, vec4(cvec, 1.0));
}