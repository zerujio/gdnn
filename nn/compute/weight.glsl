#[compute]

#version 450

const uint P_MAX = 6;

layout(local_size_x = 1 << P_MAX) in;

// Input/output size exponent.
layout(push_constant) uniform Push {
    uvec2 p;
};

layout(std430, set = 0, binding = 0)
restrict readonly buffer Weight {
    float w[];
};

layout(std430, set = 1, binding = 0)
restrict readonly buffer Input {
    float x[];
};

layout(std430, set = 2, binding = 0)
restrict writeonly buffer Output {
    float y[];
};

shared float wx[gl_WorkGroupSize.x];

void main() {
    const uint local_idx = gl_LocalInvocationID.x;
    const uint global_idx = gl_GlobalInvocationID.x;

    const uint out_idx = global_idx >> p.x; // = global_idx / size.x
    const uint instance_idx = out_idx >> p.y; // = global_idx / (size.x * size.y)

    {
        const float w_ij = w[global_idx];

        const uint i = local_idx & ~(-1 << p.x); // = local_idx % size.x
        const float x_i = x[instance_idx << p.x + i];

        wx[local_idx] = w_ij * x_i;
    }

    barrier();
    memoryBarrierShared();

    // sum up partial results
    uint mod_mask = 0;
    for (uint q = 0; q < p.x; ++q) {
        const uint offset = 1 << q;
        mod_mask |= offset;

        // if (local_idx % 2^p.x == 0)
        if ((local_idx & mod_mask) == 0) {
            wx[local_idx] += wx[local_idx + offset];
        }

        barrier();
        memoryBarrierShared();
    }

    if ((local_idx & mod_mask) == 0) {
        y[out_idx] += wx[gl_LocalInvocationID.x];
    }
}
