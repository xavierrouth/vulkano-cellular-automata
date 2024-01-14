#version 460

// CELLS_X, CELLS_Y are macros.

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

layout(set = 0, binding = 1) buffer LifeInBuffer {
    uint life_in[];
};

layout(set = 0, binding = 2) buffer LifeOutBuffer {
    uint life_out[];
};


ivec2 wrap_position(ivec2 pos) {
    ivec2 wrapped = pos;

    atomicOr(life_out[0], 0xFF);

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

    return wrapped;
}

uint get_cell(ivec2 pos) {
    ivec2 wrapped = wrap_position(pos);

    uint byte_idx = (wrapped.y * CELLS_X) + wrapped.x;
    uint word_idx = byte_idx / 4;
    uint offset_in_word = (byte_idx % 4) * 8; // Bit offset

    return (life_in[word_idx] >> offset_in_word) & 0xFF;
}

void set_cell(ivec2 pos, uint cell_value) {
    ivec2 wrapped = wrap_position(pos);

    uint byte_idx = (wrapped.y * CELLS_X) + wrapped.x;
    uint word_idx = byte_idx / 4;
    uint offset_in_word = (byte_idx % 4) * 8; // Bit offset
     
    if (cell_value == 1) {
        atomicOr(life_out[word_idx], (0x00000001 << offset_in_word));
    }
    else {
        uint mask = 0x01 << offset_in_word;
        mask = ~mask;
        atomicAnd(life_out[word_idx], mask); // Just turn off a bit.
    }
    
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy); // 0 Indexed

    // Game Of Life Rules:
    uint neighbors = 0;

    neighbors += get_cell(pos + ivec2(-1, -1));

    neighbors += get_cell(pos + ivec2(-1, 0));

    neighbors += get_cell(pos + ivec2(-1, 1));

    neighbors += get_cell(pos + ivec2(0, -1));

    neighbors += get_cell(pos + ivec2(0, 1));

    neighbors += get_cell(pos + ivec2(1, -1));

    neighbors += get_cell(pos + ivec2(1, 0));

    neighbors += get_cell(pos + ivec2(1, 1));

    // Default
    uint current_state = get_cell(pos); 
    uint next_state = current_state;

    if (current_state == 1) {
        if (neighbors < 2) {
            next_state = 0;
        }
        if (neighbors > 3) {
            next_state = 0;
        }
    }
    else if (current_state == 0) {
        if (neighbors == 3) {
            next_state = 1;
        }
        else {
            next_state = 0;
        }
    }

    /* 
    if (pos.y % 2 == 1) {
        set_cell(pos, 1); 
    }
    else {
        set_cell(pos, 0); 
    } */
    set_cell(pos, next_state);
    

    vec3 cvec;

    if (current_state == 0) {
        cvec = vec3(0.0, 0.0, 0.0);
    }
    else {
        cvec = vec3(1.0, 1.0, 1.0);
    }

    imageStore(img, pos, vec4(cvec, 1.0));
}