extends Node

const GROUP_SIZE := Vector3i(32, 1, 1)
const SHADER_FILE: RDShaderFile = preload("res://compute/layer.glsl")

@export_range(1, 1024) var input_size := 8
@export_range(1, 1024) var output_size := 4

var rd := RenderingServer.create_local_rendering_device()
var _pipeline: RID
var _input_buf: RID
var _input_set: RID
var _layer_buf: RID
var _layer_set: RID
var _output_buf: RID
var _output_set: RID


func _ready() -> void:
	var shader := rd.shader_create_from_spirv(SHADER_FILE.get_spirv())
	_pipeline = rd.compute_pipeline_create(shader)
	
	var uniform := RDUniform.new()
	uniform.binding = 0
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	
	# input
	_input_buf = rd.storage_buffer_create(input_size * 4)
	uniform.add_id(_input_buf)
	_input_set = rd.uniform_set_create([uniform], shader, 0)
	uniform.clear_ids()
	
	# bias and weight
	_layer_buf = rd.storage_buffer_create((input_size + 1) * output_size * 4)
	uniform.add_id(_layer_buf)
	_layer_set = rd.uniform_set_create([uniform], shader, 1)
	uniform.clear_ids()
	
	# output
	_output_buf = rd.storage_buffer_create(output_size * 4)
	uniform.add_id(_output_buf)
	_output_set = rd.uniform_set_create([uniform], shader, 2)


func set_weight_and_bias(weight: PackedFloat32Array, bias: PackedFloat32Array) -> void:
	var buf := bias.to_byte_array()
	buf.append_array(weight.to_byte_array())
	rd.buffer_update(_layer_buf, 0, buf.size(), buf)


func set_input(data: PackedFloat32Array) -> void:
	assert(data.size() == input_size)
	rd.buffer_update(_input_buf, 0, data.size() * 4, data.to_byte_array())


func dispatch() -> PackedFloat32Array:
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, _pipeline)
	rd.compute_list_bind_uniform_set(cl, _input_set, 0)
	rd.compute_list_bind_uniform_set(cl, _layer_set, 1)
	rd.compute_list_bind_uniform_set(cl, _output_set, 2)
	@warning_ignore("integer_division")
	rd.compute_list_dispatch(cl, 1 + output_size / GROUP_SIZE.x, 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	return rd.buffer_get_data(_output_buf).to_float32_array()
