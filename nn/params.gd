class_name NNParams
extends Resource
## Stores the parameters of a neural network.

## Layout of the network. Specifies the inputs, layers, and outputs.
## Changing the layout will discard any stored data.
@export
var layout: NNLayout:
	set(l):
		if layout:
			layout.changed.disconnect(_layout_changed)
		layout = l
		if layout:
			layout.changed.connect(_layout_changed)
		_layout_changed()
## Number of instances stored in this parameter set.
## Changing this value will reallocate memory.
@export
var count := 0:
	set = _set_count

var _ctx: NNContext = GlobalNNContext
var _buffers: Array[NNContext.ArrayBuffer]


func update_layer_weights(idx: int, data: PackedByteArray) -> void:
	var buf := _weight_buffer(idx)
	assert(buf.size_bytes() == data.size())
	buf.update(0, data)



func update_layer_bias(idx: int, data: PackedByteArray) -> void:
	var buf := _bias_buffer(idx)
	assert(buf.size_bytes() == data.size())
	buf.update(0, data)


func _layout_changed() -> void:
	if not layout:
		_buffers.clear()
		return
	
	var layer_count := layout.get_layer_count()
	var old_buffer_size := _buffers.size()
	_buffers.resize(2 * layer_count)
	for i in range(old_buffer_size, _buffers.size()):
		_buffers[i] = NNContext.ArrayBuffer.new(_ctx)
	
	if count < 1:
		return
	
	var size := Vector2i(layout.get_input_size(), 0)
	for i in range(layer_count):
		size.y = layout.get_layer_output_size(i)
		_weight_buffer(i).allocate_matrix(count, size)
		_bias_buffer(i).allocate_vector(count, size.y)
		size.x = size.y
	
	emit_changed()


func _set_count(n: int) -> void:
	if count == n:
		return
	
	var size := Vector2i(layout.get_input_size(), 0)
	for i in range(layout.get_layer_count()):
		size.y = layout.get_layer_output_size(i)
		_weight_buffer(i).resize_matrix(n, size)
		_bias_buffer(i).resize_vector(n, size.y)
		size.x = size.y
	
	count = n
	emit_changed()


func _weight_buffer(idx: int) -> NNContext.ArrayBuffer:
	return _buffers[2 * idx]


func _bias_buffer(idx: int) -> NNContext.ArrayBuffer:
	return _buffers[2 * idx + 1]
