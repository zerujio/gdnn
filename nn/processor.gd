class_name NNProcessor
extends Node


## Input size exponent. 
@export_range(0, NNContext.INPUT_EXP_MAX) var input_exp: int:
	set(p): 
		context.input_exp = p
		if _in:
			_in.resize_data_buffer()
	get: return context.input_exp
@export_range(0, NNContext.INPUT_EXP_MAX) var output_exp: int:
	set(p): context.output_exp = p
	get: return context.output_exp
@export var activation: NNContext.Activation:
	set(a): context.activation = a
	get: return context.activation

var context := NNContext.new()
var _instance_count: int = 0:
	set(n):
		assert(n >= 0)
		_in.resize(n * get_input_count())
		context.set_count(n)
	get: return context.instance_count
var _free_list: PackedInt32Array
var _in: NNInput
var _out: NNOutput


func _ready() -> void:
	_out = NNOutput.new()
	_out.ctx = context
	_out.process_physics_priority = process_physics_priority - 1
	add_child(_out, false, Node.INTERNAL_MODE_FRONT)
	
	_in = NNInput.new()
	_in.ctx = context
	_in.process_physics_priority = process_physics_priority + 1
	add_child(_in, false, Node.INTERNAL_MODE_FRONT)


## Allocates a new instance and returns its index.
func add_instance(params := PackedFloat32Array()) -> int:
	var idx: int
	
	var w_count := get_weight_count()
	var w: PackedByteArray
	var b: PackedByteArray
	
	if not params.is_empty():
		w = params.slice(0, w_count).to_byte_array()
		b = params.slice(w_count).to_byte_array()
	
	if _free_list.is_empty():
		idx = _instance_count
		context.add_instances(1, w, b)
	else:
		var free_idx := _free_list.size() - 1
		idx = _free_list[free_idx]
		_free_list.remove_at(free_idx)
		
		if not w.is_empty():
			context.update_weight(idx, 1, w)
		
		if not b.is_empty():
			context.update_bias(idx, 1, b)
	
	return idx


## Removes an instance.
func remove_instance(idx: int) -> void:
	assert(is_idx_valid(idx))
	_free_list.push_back(idx)


## Sets an instance's input
func set_instance_input(instance_idx: int, data: PackedFloat32Array) -> void:
	assert(is_idx_valid(instance_idx))
	assert(data.size() == get_input_count(), 
		"mismatched input count. Got %d, expected %d" % [data.size(), get_input_count()])
	
	_in.data.resize(context.input_count * context.instance_count * 4)
	
	var offset := data.size() * instance_idx * 4
	for x in data:
		_in.data.encode_float(offset, x)
		offset += 4
	
	_in.dirty = true
	_out.dirty = true


## Read an instance's output.
func get_instance_output(instance_idx: int) -> PackedFloat32Array:
	assert(is_idx_valid(instance_idx))
	var size := get_output_count() * _instance_count * 4
	return _out.data.slice(instance_idx * size, (instance_idx + 1) * size).to_float32_array()


## Updates an instance's weights and biases.
func set_instance_params(instance_idx: int, params: PackedFloat32Array) -> void:
	assert(is_idx_valid(instance_idx))
	var w_count := get_weight_count()
	context.update_weight(instance_idx, 1, params.slice(0, w_count).to_byte_array())
	context.update_bias(instance_idx, 1, params.slice(w_count).to_byte_array())


func is_idx_valid(idx: int) -> bool:
	return idx >= 0 and idx < _instance_count and not _free_list.has(idx)


func get_input_count() -> int:
	return context.input_count


func get_output_count() -> int:
	return context.output_count


func get_weight_count() -> int:
	return get_input_count() * get_output_count()


class NNInput:
	extends Node
	
	var ctx: NNContext
	var data: PackedByteArray
	var dirty := false
	
	
	func _ready() -> void:
		resize_data_buffer()
	
	
	func _physics_process(_delta: float) -> void:
		if dirty:
			ctx.submit_input(data)
			dirty = false
	
	
	func resize_data_buffer() -> void:
		data.resize(ctx.input_count * ctx.instance_count * 4)
		print_debug(data.size())


class NNOutput:
	extends Node
	
	var ctx: NNContext
	var data: PackedByteArray
	var dirty := false
	
	
	func _ready() -> void:
		data.resize(ctx.output_count * ctx.instance_count * 4)
	
	
	func _physics_process(_delta: float) -> void:
		if dirty:
			data = ctx.sync_output()
			dirty = false
