#[compute]
#version 450

layout(local_size_x = 64) in;

const uint OP_LERP = 0;

layout(constant_id = 0)
const uint op_index = 0;

layout(push_constant)
uniform PushConsant {
    uint size_log2;
};

layout(std430, set = 0, binding = 0)
readonly buffer A {
    float a[];
};

layout(std430, set = 1, binding = 0)
readonly buffer B {
    float b[];
};

layout(std430, set = 2, binding = 0)
restrict readonly buffer Param {
    float params[];
};

layout(std430, set = 3, binding = 0)
restrict readonly buffer Index {
    uvec2 indices[];
};

layout(std430, set = 4, binding = 0)
restrict writeonly buffer Result {
    float result[];
};

float lerp(float x, float y, float a) {
    return x * a + y * (1.0 - a);
}

float op(vec2 v, float p) {
    switch (op_index) {
        case OP_LERP:
        return lerp(v.x, v.y, p);
    }
}

void main() {
    const uint idx = gl_GlobalInvocationID.x;
    if (idx >= result.length()) {
        return;
    }

    const uint group_idx = idx >> size_log2; // idx / size
    const uvec2 pair_idx = indices[group_idx];
    const uvec2 v_idx = (pair_idx << size_log2) + (idx & ~(-1 << size_log2)); // pair_idx * size + idx % size
    const vec2 v = vec2(a[v_idx.x], b[v_idx.y]);

    result[idx] = op(v, params[idx]);
}
