#version 460

// CELLS_X, CELLS_Y are macros.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

layout(set = 0, binding = 1) readonly buffer LifeInBuffer {
    uint life_in[];
};

layout(set = 0, binding = 2) writeonly buffer LifeOutBuffer {
    uint life_out[];
};


uint get_idx(ivec2 pos) {
    ivec2 wrapped = pos;

    if (pos.x >= CELLS_X) { // 0 indexed.
        wrapped.x = 0;
    }
    else if (pos.x < 0) {
        wrapped.x = CELLS_X - 1;
    }
    if (pos.y >= CELLS_Y) {
        wrapped.y = 0;
    }
    else if (pos.y < 0) {
        wrapped.y = CELLS_Y - 1;
    }

    return (wrapped.y * CELLS_X) + wrapped.x;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy); // 0 Indexed

    uint idx = get_idx(pos);
    uint out_state = 0;

    // Game Of Life Rules:
    uint neighbors = 0;

    neighbors += life_in[get_idx(pos + ivec2(-1, -1))];

    neighbors += life_in[get_idx(pos + ivec2(-1, 0))];

    neighbors += life_in[get_idx(pos + ivec2(-1, 1))];

    neighbors += life_in[get_idx(pos + ivec2(0, -1))];

    neighbors += life_in[get_idx(pos + ivec2(0, 1))];

    neighbors += life_in[get_idx(pos + ivec2(1, -1))];

    neighbors += life_in[get_idx(pos + ivec2(1, 0))];

    neighbors += life_in[get_idx(pos + ivec2(1, 1))];

    // Default
    life_out[idx] = life_in[idx];
    uint x = life_in[idx];

    if (life_in[idx] == 1) {
        if (neighbors < 2) {
            x = 0;
            life_out[idx] = 0;
        }
        if (neighbors > 3) {
            x = 0;
            life_out[idx] = 0;
        }
    }

    else if (life_in[idx] == 0) {
        if (neighbors == 3) {
            life_out[idx] = 1;
            x = 1;
        }
    }


    vec3 cvec;

    if (x == 0) {
        cvec = vec3(0.0, 0.0, 0.0);
    }
    else {
        cvec = vec3(1.0, 1.0, 1.0);
    }

    imageStore(img, pos, vec4(cvec, 1.0));
}