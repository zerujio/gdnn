@abstract
class_name NNParams
extends Resource
## Stores the parameters of a neural network.

## Layout of the network. Specifies the inputs, layers, and outputs.
@export var layout: NNLayout

enum Type { WEIGHT, BIAS }

## Copies a set of parameters to a byte array.
@abstract func copy_to_buffer(layer_idx: int, param_type: Type, buffer: PackedByteArray) -> void
