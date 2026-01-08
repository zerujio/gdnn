#[compute]
#version 450

layout(local_size_x = 64) in;

const uint OP_ADD = 0;

layout(constant_id = 0)
const uint op_index = 0;

layout(std430, set = 0, binding = 0)
readonly buffer X {
    float x[];
};

layout(std430, set = 1, binding = 0)
readonly buffer Y {
    float y[];
};

layout(std430, set = 2, binding = 0)
writeonly buffer Result {
    float result[];
};

float op2(vec2 v) {
    switch (op_index) {
        case OP_ADD:
        return v.x + v.y;
    }
}

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i < result.length()) {
        result[i] = op2(vec2(x[i], y[i]));
    }
}
