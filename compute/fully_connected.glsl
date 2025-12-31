#[compute]
#version 450

layout(local_size_x = 32) in;

// array of size N
layout(set = 0, binding = 0) restrict readonly
buffer b_input {
    float x[];
};

// 2D array of size (1 + N) x M.
// 1 bias row followed by N weight rows.
layout(set = 1, binding = 0) restrict readonly
buffer b_layer {
    float bw[];
};

// array of size M
layout(set = 2, binding = 0) restrict writeonly
buffer b_output {
    float y[];
};

// j: output index
float bias(uint j) {
    return bw[j];
}

// i: input index
// j: output index
float weight(uint i, uint j) {
    return bw[(1 + i) * y.length() + j];
}

void main() {
    const uint j = gl_GlobalInvocationID.x;

    if (j >= y.length()) {
        return;
    }

    // y = X * W + b
    y[j] = bias(j); // + b
    for (uint i = 0; i < x.length(); ++i) {
        y[j] += x[i] * weight(i, j); // + X * W
    }
}
