#[compute]
#version 450

layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0)
readonly buffer X {
    float x[];
};

layout(std430, set = 1, binding = 0)
writeonly buffer Y {
    float y[];
};

// values:
// 0: sigmoid
// 1: ReLU
layout(constant_id = 0) const uint ACTIVATION_TYPE = 0;

float sigmoid(float z) {
    return 1.0 / (1.0 + exp(-z));
}

float relu(float z) {
    return max(0, z);
}

float activation(float z) {
    switch (ACTIVATION_TYPE) {
        case 0:
        return sigmoid(z);

        case 1:
        return relu(z);

        default:
        return 0.0f;
    }
}

void main() {
    const uint i = gl_GlobalInvocationID.x;
    y[i] = activation(x[i]);
}
