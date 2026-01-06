class_name NNMultiInstance
extends Node
## Contains multiple neural network instances (parameter sets) with the same params.

## Neural network parameters.
##
## [b]Warning:[/b] changing this resource will discard all instance data.
@export var params: NNParams:
	set(p):
		if params: 
			params.changed.disconnect(_params_changed)
		params = p
		if params:
			params.changed.connect(_params_changed)
		_params_changed()

## Input buffer.
var _input_buf := NNContext.ArrayBuffer.new(_ctx())
## Layer output buffers.
var _output_buf: Array[NNContext.ArrayBuffer]

## Cached input/output
var _input_cache: PackedByteArray
var _output_cache: PackedByteArray


## Updates the input data for one or more instances starting at [param instance_idx].
## Number of instances updated depends on size of [param data].
func set_input(instance_idx: int, data: PackedFloat32Array) -> void:
	assert(instance_idx >= -instance_count() and instance_idx < instance_count())
	var input_size := params.layout.get_input_size()
	assert(data.size() % input_size == 0)
	const FLOAT_SIZE := 4
	var offset := instance_idx * FLOAT_SIZE
	for x in data:
		_input_cache.encode_float(offset, x)
		offset += FLOAT_SIZE


## Submits input data for processing.
func submit_input() -> void:
	assert(params and params.layout)
	var lc := params.layout.get_layer_count() if params and params.layout else 0
	if lc < 1: 
		_output_cache = _input_cache.duplicate()
		return
	
	_input_buf.update(0, _input_cache)
	
	var ctx := _ctx()
	var input := _input_buf
	var size_log2 := Vector2i(params.layout.input_log2, 0)
	for i in range(lc):
		size_log2.y = params.layout.get_layer_output_log2(i)
		var output := _output_buf[i]
		var weight := params._weight_buffer(i)
		var bias := params._bias_buffer(i)
		NNContext.ArrayBuffer.copy_vector(bias, output, 0, 0, params.count, 2 ** size_log2.y)
		ctx.matrix_vector_multiply_add(weight, size_log2, input, output)
		
		match params.layout.get_layer_activation(i):
			NNLayout.Activation.SIGMOID:
				ctx.sigmoid(output, output)
			NNLayout.Activation.RELU:
				ctx.relu(output, output)
		
		input = output
		size_log2.x = size_log2.y
	
	input.get_data_async(func(d: PackedByteArray):
		_output_cache = d)


## Retrieves the output data of one or more instances.
## [b]Note[/b]: this accesses cached output data.
func get_output(instance_idx: int, count := 1) -> PackedFloat32Array:
	assert(instance_idx >= 0)
	assert(count > 0)
	assert(instance_idx + count <= instance_count())
	var stride := params.layout.get_output_size() * 4 if params and params.layout else 0
	var begin := instance_idx * stride
	var end := begin + count * stride
	return _output_cache.slice(begin, end).to_float32_array()


func instance_count() -> int:
	return params.count if params else 0


func _params_changed() -> void:
	if not params or not params.layout:
		_input_buf.allocate(0)
		_output_buf.clear()
		_input_cache.clear()
		_output_cache.clear()
		return
	
	var old_layer_count := _output_buf.size()
	var layer_count := params.layout.get_layer_count()
	
	_output_buf.resize(layer_count)
	for i in range(old_layer_count, layer_count):
		_output_buf[i] = NNContext.ArrayBuffer.new(_ctx())
	
	_input_buf.allocate_vector(params.count, params.layout.get_input_size())
	_input_cache.resize(_input_buf.size_bytes())
	
	for i in range(layer_count):
		_output_buf[i].allocate_vector(params.count, params.layout.get_layer_output_size(i))
	
	var output_buf := _output_buf[-1] if layer_count > 0 else _input_buf
	_output_cache.resize(output_buf.size_bytes())


func _ctx() -> NNContext: 
	return GlobalNNContext
