#[compute]
#version 450
// element-wise ternary operations.
const uint OP_LERP = 0;

layout(constant_id = 0) const uint op_index = 0;

layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0)
readonly buffer X {
    float x[];
};

layout(std430, set = 1, binding = 0)
readonly buffer Y {
    float y[];
};

layout(std430, set = 2, binding = 0)
readonly buffer Z {
    float z[];
};

layout(std430, set = 3, binding = 0)
writeonly buffer Result {
    float result[];
};

float op3(vec3 v) {
    switch (op_index) {
        case OP_LERP:
        return v.x * v.z + v.y * (1.0 - v.z);
    }
}

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i < result.length()) {
        result[i] = op3(vec3(x[i], y[i], z[i]));
    }
}
