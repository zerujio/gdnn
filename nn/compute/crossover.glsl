#[compute]
#version 450

layout(local_size_x = 64) in;

const uint TYPE_INTERMEDIATE = 0;

// crossover function type
layout(constant_id = 0)
const uint TYPE = 0;

struct Crossover {
    // parent indices
    uvec2 parent_id;
    // crossover parameters
    uvec2 param;
};

layout(std430, set = 0, binding = 0)
restrict readonly buffer ParentBuffer {
    float parents[];
};

layout(std430, set = 1, binding = 0)
restrict writeonly buffer ChildBuffer {
    float children[];
};

layout(std430, set = 2, binding = 0)
restrict readonly buffer CrossoverBuffer {
    Crossover crossovers[];
};

float intermediate_crossover(vec2 parent, float b) {
    return parent.x * b + parent.y * (1.0 - b);
}

void main() {
    const uint i = gl_GlobalInvocationID.x;
    const Crossover cross = crossovers[i];
    const vec2 parent = {
            parents[cross.parent_id.x],
            parents[cross.parent_id.y]
        };

    float child;
    switch (TYPE) {
        case TYPE_INTERMEDIATE:
        child = intermediate_crossover(parent, uintBitsToFloat(cross.param.x));
        break;
    }

    children[i] = child;
}
