class_name NNLayout
extends Resource
## Specifies the number, size and type of layers in a neural network.
## 
## [b]Note[/b]: [signal changed] is NOT emitted automatically when properties changed. After a 
## change, call [method emit_changed] to update any dependent [class NNMultiInstance] nodes.

## Activation function types.
enum Activation { NONE, SIGMOID, RELU }

## Base 2 logarithm of the input size. This is ensures that the input size is
## always a power of 2.
@export_range(0, NNContext.INPUT_EXP_MAX) var input_log2: int = 0

## Stores the layer definitions. Each layer takes up 2 bytes, the first of which
## is the [enum Activation] function, and the second is the log2 of the output size.
@export var layer_data: PackedByteArray


func add_layer(activation: Activation, output_size_log2: int) -> void:
	assert(output_size_log2 >= 0 and output_size_log2 < NNContext.INPUT_EXP_MAX)
	layer_data.append(activation)
	layer_data.append(output_size_log2)


func remove_layer(idx: int) -> void:
	assert(idx < get_layer_count())
	assert(idx >= -get_layer_count())
	layer_data.remove_at(2 * idx + 1)
	layer_data.remove_at(2 * idx)


## Number of layers.
func get_layer_count() -> int:
	assert(layer_data.size() % 3 == 0)
	@warning_ignore("integer_division")
	return layer_data.size() / 2


## Number of inputs of the first layer.
func get_input_size() -> int:
	return 2 ** input_log2


## Number of outputs of the last layer.
func get_output_size() -> int:
	return 2 ** get_output_size_log2()


## Base 2 logarithm of the number of outputs of the last layer.
func get_output_size_log2() -> int:
	return layer_data[-1] if not layer_data.is_empty() else get_input_size()


func get_layer_activation(idx: int) -> Activation:
	return layer_data[idx * 2] as Activation


func get_layer_output_log2(idx: int) -> int:
	var log2 := layer_data[idx * 2 + 1]
	assert(log2 > 0, "layer with output log2 = %d < 1" % log2)
	return log2


func get_layer_output_size(idx: int) -> int:
	return 2 ** get_layer_output_log2(idx)


func get_layer_input_size(idx: int) -> int:
	return get_layer_output_size(idx - 1) if idx > 0 else get_input_size()


## Returns Vector2i containing the input and output size of a layer.
func get_layer_size(idx: int) -> Vector2i:
	return Vector2i(get_layer_input_size(idx), get_layer_output_size(idx))
