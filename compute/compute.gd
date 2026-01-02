extends Node

const WEIGHT_SHADER_FILE: RDShaderFile = preload("res://compute/weight.glsl")
const ACTIVATION_SHADER_FILE: RDShaderFile = preload("res://compute/activation.glsl")
const INPUT_EXP_MAX := 6
const FLOAT_SIZE := 4
const GROUP_SIZE := 2 ** INPUT_EXP_MAX

## Input size exponent. 
@export_range(1, INPUT_EXP_MAX) var input_exp := 3
@export_range(1, INPUT_EXP_MAX) var output_exp := 2
## Number of instances stored/active.
@export_range(0, 1024) var instance_count := 0:
	set = set_count

## Number of instances allocated.
var instance_capacity := 0:
	set = set_capacity
var input_count: int:
	get: return 2 ** input_exp
var output_count: int:
	get: return 2 ** output_exp

var rd := RenderingServer.create_local_rendering_device()

## Multiplies input by layer weights.
var _weight_shader: RID = rd.shader_create_from_spirv(WEIGHT_SHADER_FILE.get_spirv(), "weight")
var _weight_pipeline: RID = rd.compute_pipeline_create(_weight_shader)

## Applies activation functions.
var _activation_shader: RID = rd.shader_create_from_spirv(ACTIVATION_SHADER_FILE.get_spirv(), "activation")
var _sigmoid_pipeline: RID
var _relu_pipeline: RID

var _input_buf: RID
var _input_set: RID

var _output_buf: RID
var _output_set: RID

var _weight_buf: RID
var _weight_set: RID
var _bias_buf: RID


func _init() -> void:
	# activation functions
	var spec_const := RDPipelineSpecializationConstant.new()
	
	# sigmoid
	spec_const.value = 0
	_sigmoid_pipeline = rd.compute_pipeline_create(_activation_shader, [spec_const])
	
	# relu
	spec_const.value = 1
	_relu_pipeline = rd.compute_pipeline_create(_activation_shader, [spec_const])


func set_count(count: int) -> void:
	assert(count >= 0)
	
	if count > instance_capacity:
		var new_cap := 1
		while new_cap < count:
			new_cap *= 2
		instance_capacity = new_cap
	
	instance_count = count


func set_capacity(capacity: int) -> void:
	assert(capacity >= 0)
	
	var weight_stride := FLOAT_SIZE * input_count * output_count
	_weight_buf = _resize_storage_buffer(_weight_buf, 
		weight_stride * instance_capacity, 
		weight_stride * capacity)
	
	var bias_stride := FLOAT_SIZE * output_count
	_bias_buf = _resize_storage_buffer(_bias_buf, 
		bias_stride * instance_capacity, 
		bias_stride * capacity)
	
	var input_stride := FLOAT_SIZE * input_count
	_input_buf = _resize_storage_buffer(_input_buf, 
		input_stride * instance_capacity, 
		input_stride * capacity)
	
	var output_stride := FLOAT_SIZE * output_count
	_output_buf = _resize_storage_buffer(_output_buf,
		output_stride * instance_capacity,
		output_stride * capacity)
	
	if capacity > 0:
		assert(not rd.uniform_set_is_valid(_weight_set))
		_weight_set = rd.uniform_set_create(
			[_storage_buffer_uniform(_weight_buf, 0)], 
			_weight_shader, 
			0)
	
		assert(not rd.uniform_set_is_valid(_input_set))
		_input_set = rd.uniform_set_create(
			[_storage_buffer_uniform(_input_buf, 0)], 
			_weight_shader, 
			1)
		
		assert(not rd.uniform_set_is_valid(_output_set))
		_output_set = rd.uniform_set_create(
			[_storage_buffer_uniform(_output_buf, 0)],
			_weight_shader,
			2)
	
	instance_capacity = capacity


func add_instances(count: int, weight_data: PackedByteArray, bias_data: PackedByteArray) -> void:
	assert(count > 0)
	
	var old_count := instance_count
	instance_count += count
	
	update_weight(old_count, count, weight_data)
	update_bias(old_count, count, bias_data)


func update_weight(index: int, count: int, data: PackedByteArray) -> void:
	assert(index >= 0)
	assert(count >= 0)
	var stride := FLOAT_SIZE * input_count * output_count
	assert(data.size() == count * stride)
	rd.buffer_update(_weight_buf, index * stride, data.size(), data)


func update_bias(index: int, count: int, data: PackedByteArray) -> void:
	assert(index >= 0)
	assert(count >= 0)
	var stride := FLOAT_SIZE * output_count
	assert(data.size() == count * stride)
	rd.buffer_update(_bias_buf, index * stride, data.size(), data)


func set_input(input_data: PackedByteArray) -> void:
	assert(input_data.size() == FLOAT_SIZE * input_count * instance_count)
	rd.buffer_update(_input_buf, 0, input_data.size(), input_data)


func submit_input(input_data := PackedByteArray()) -> void:
	if not input_data.is_empty():
		set_input(input_data)
	
	# apply bias
	rd.buffer_copy(_bias_buf, _output_buf, 0, 0, FLOAT_SIZE * output_count * instance_count)
	
	var cl := rd.compute_list_begin()
	
	# apply weights to inputs
	var push_constant: PackedByteArray
	push_constant.resize(16)
	push_constant.encode_u32(0, input_exp)
	push_constant.encode_u32(4, output_exp)
	
	rd.compute_list_bind_compute_pipeline(cl, _weight_pipeline)
	rd.compute_list_set_push_constant(cl, push_constant, push_constant.size())
	rd.compute_list_bind_uniform_set(cl, _weight_set, 0)
	rd.compute_list_bind_uniform_set(cl, _input_set, 1)
	rd.compute_list_bind_uniform_set(cl, _output_set, 2)
	@warning_ignore("integer_division")
	rd.compute_list_dispatch(cl, 1 + instance_count * input_count * output_count / GROUP_SIZE, 1, 1)
	
	rd.compute_list_add_barrier(cl)
	
	# apply activation function
	rd.compute_list_bind_compute_pipeline(cl, _relu_pipeline)
	rd.compute_list_bind_uniform_set(cl, _output_set, 0)
	@warning_ignore("integer_division")
	rd.compute_list_dispatch(cl, 1 + instance_count * output_count / GROUP_SIZE, 1, 1)
	
	rd.compute_list_end()
	rd.submit()


func sync_output() -> PackedByteArray:
	rd.sync()
	return rd.buffer_get_data(_output_buf, 0, FLOAT_SIZE * output_count * instance_count)


func _resize_storage_buffer(old_buffer: RID, old_size: int, new_size: int) -> RID:
	var new_buffer: RID
	
	if new_size > 0:
		new_buffer = rd.storage_buffer_create(new_size)
	
	if old_buffer:
		assert(old_size > 0)
		if new_buffer:
			rd.buffer_copy(old_buffer, new_buffer, 0, 0, old_size)
		rd.free_rid(old_buffer)
	
	return new_buffer


func _storage_buffer_uniform(buffer: RID, binding: int) -> RDUniform:
	var uniform := RDUniform.new()
	uniform.binding = binding
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.add_id(buffer)
	return uniform
