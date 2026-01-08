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


## Copies parameter data between two parameter sets with compatible layouts.
func copy_layer_weights_to(other: NNParams,
		from_layer: int, to_layer: int, 
		from_instance := 0, to_instance := 0,
		instance_count := 1) -> Error:
	assert(layout)
	assert(other and other.layout)
	var from_buf := _weight_buffer(from_layer)
	var to_buf := other._weight_buffer(to_layer)
	var layer_size := layout.get_layer_size(from_layer)
	assert(layer_size == other.layout.get_layer_size(to_layer))
	return from_buf.copy_matrix_to(to_buf, from_instance, to_instance, instance_count, layer_size)


func copy_layer_bias_to(other: NNParams, 
		from_layer: int, to_layer: int,
		from_instance := 0, to_instance := 0, 
		instance_count := 1) -> Error:
	assert(layout)
	assert(other and other.layout)
	var from_buf := _bias_buffer(from_layer)
	var to_buf := other._weight_buffer(to_layer)
	var output_size := layout.get_layer_output_size(from_layer)
	assert(output_size == other.layout.get_layer_output_size(to_layer))
	return from_buf.copy_vector_to(to_buf, from_instance, to_instance, instance_count, output_size)


func copy_layer_to(other: NNParams, from_layer: int, to_layer: int,
		from_instance := 0, to_instance := 0,
		instance_count := 1) -> Error:
	var err := copy_layer_weights_to(other, from_layer, to_layer, from_instance, to_instance, 
		instance_count)
	if not err:
		err = copy_layer_bias_to(other, from_layer, to_layer, from_instance, to_instance, 
			instance_count)
	return err


## Creates a new parameter set by interpolating randomly between pairs of instances.
func intermediate_crossover(pairs: PackedInt32Array, d := 0.25) -> NNParams:
	var new_params := NNParams.new()
	new_params.layout = layout
	
	if pairs.is_empty():
		return new_params
	
	assert(layout)
	assert(pairs.size() % 2 == 0)
	assert(Array(pairs).all(func(idx: int): return idx >= 0 and idx < count))
	
	new_params.count = pairs.size() >> 1 # = pairs.size() / 2
	
	var pair_buf := _ctx.create_array_buffer(pairs.size(), pairs.to_byte_array())
	var a: PackedByteArray # interpolation weights
	var size_log2 := Vector2i(layout.input_log2, 0)
	for i in range(layout.get_layer_count()):
		size_log2.y = layout.get_layer_output_log2(i)
		var size := Vector2i(2 ** size_log2.x, 2 ** size_log2.y)
		
		# interpolate weights, then biases
		for j in range(2):
			var old := _buffers[2 * i + j]
			var new := new_params._buffers[2 * i + j]
			
			a.resize(new.size_bytes())
			for k in range(0, a.size(), 4):
				# random interpolation weight in range [-d, 1 + d]
				a.encode_float(k, randf_range(-d, 1.0 + d))
			var a_buf := _ctx.create_array_buffer(new.size, a)
			
			_ctx.interpolate_indexed(old, old, a_buf, pair_buf, size_log2.x * size_log2.y, new)
		
		size_log2.x = size_log2.y
	
	return new_params


## Adds random values to the parameter data.
func randomize_params(range_min := -1.0, range_max := 1.0) -> void:
	for i in range(layout.get_layer_count() if layout else 0):
		randomize_layer_params(i, range_min, range_max)


func randomize_layer_params(layer_idx: int, range_min := -1.0, range_max := 1.0) -> void:
	var r: PackedByteArray
	for i in range(2):
		var buf := _buffers[layer_idx + i]
		r.resize(buf.size_bytes())
		for j in range(0, r.size(), 4):
			r.encode_float(j, randf_range(range_min, range_max))
		var rand_buf := _ctx.create_array_buffer(buf.size, r)
		_ctx.add(buf, rand_buf, buf)


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
		var w := _weight_buffer(i)
		w.allocate_matrix(count, size)
		w.clear()
		var b := _bias_buffer(i)
		b.allocate_vector(count, size.y)
		b.clear()
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
