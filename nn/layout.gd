class_name NNLayout
extends Resource
## Specifies the number, size and type of layers in a neural network.

## Layer types.
enum Layer { FULLY_CONNECTED }

## Activation function types.
enum Activation { NONE, SIGMOID, RELU }

## Size of the input layer.
@export_range(1, 64, 1, "exp") var input_size: int

## Stores the layer definitions. Each layer takes up 3 bytes, with the following meaning:
## 0: [enum Layer] type
## 1: [enum Activation] function
## 2: output size
@export var layer_data: PackedByteArray


## Number of layers.
func get_layer_count() -> int:
	assert(layer_data.size() % 3 == 0)
	@warning_ignore("integer_division")
	return layer_data.size() / 3


## Number of outputs of the last layer.
func get_output_size() -> int:
	return layer_data[-1] if not layer_data.is_empty() else input_size


func get_layer_type(idx: int) -> Layer:
	return layer_data[idx * 3] as Layer


func get_layer_activation(idx: int) -> Activation:
	return layer_data[idx * 3 + 1] as Activation


func get_layer_output_size(idx: int) -> int:
	var size := layer_data[idx * 3 + 2]
	assert(size > 0, "layer with output size = %d < 1" % size)
	return size


func get_layer_input_size(idx: int) -> int:
	return get_layer_output_size(idx - 1) if idx > 0 else input_size
