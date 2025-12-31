extends Node

const GROUP_SIZE := Vector3i(32, 1, 1)
const FC_SHADER_FILE: RDShaderFile = preload("res://compute/fully_connected.glsl")
const ACTIVATION_SHADER_FILE: RDShaderFile = preload("res://compute/activation.glsl")

@export_range(1, 1024) var input_size := 8
@export_range(1, 1024) var output_size := 4

var rd := RenderingServer.create_local_rendering_device()

var _fc_shader: RID
var _fc_pipeline: RID
var _activation_shader: RID
var _sigmoid_pipeline: RID
var _relu_pipeline: RID

var _input_buf: RID
var _input_set: RID
var _layer_buf: RID
var _layer_set: RID
var _output_buf: RID
var _output_set: RID


func _ready() -> void:
	# fully connected layer
	_fc_shader = rd.shader_create_from_spirv(FC_SHADER_FILE.get_spirv(), "fully_connected")
	_fc_pipeline = rd.compute_pipeline_create(_fc_shader)
	
	# activation functions
	var spec_const := RDPipelineSpecializationConstant.new()
	_activation_shader = rd.shader_create_from_spirv(ACTIVATION_SHADER_FILE.get_spirv(), "activation")
	
	# sigmoid
	spec_const.value = 0
	_sigmoid_pipeline = rd.compute_pipeline_create(_activation_shader, [spec_const])
	
	# relu
	spec_const.value = 1
	_relu_pipeline = rd.compute_pipeline_create(_activation_shader, [spec_const])

	# storage buffers
	var uniform := RDUniform.new()
	uniform.binding = 0
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER

	# input
	_input_buf = rd.storage_buffer_create(input_size * 4)
	uniform.add_id(_input_buf)
	_input_set = rd.uniform_set_create([uniform], _fc_shader, 0)
	uniform.clear_ids()

	# bias and weight
	_layer_buf = rd.storage_buffer_create((input_size + 1) * output_size * 4)
	uniform.add_id(_layer_buf)
	_layer_set = rd.uniform_set_create([uniform], _fc_shader, 1)
	uniform.clear_ids()

	# output
	_output_buf = rd.storage_buffer_create(output_size * 4)
	uniform.add_id(_output_buf)
	_output_set = rd.uniform_set_create([uniform], _fc_shader, 2)


func set_weight_and_bias(weight: PackedFloat32Array, bias: PackedFloat32Array) -> void:
	var buf := bias.to_byte_array()
	buf.append_array(weight.to_byte_array())
	rd.buffer_update(_layer_buf, 0, buf.size(), buf)


func set_input(data: PackedFloat32Array) -> void:
	assert(data.size() == input_size)
	rd.buffer_update(_input_buf, 0, data.size() * 4, data.to_byte_array())


func dispatch() -> PackedFloat32Array:
	@warning_ignore("integer_division")
	var groups := 1 + output_size / GROUP_SIZE.x
	
	var cl := rd.compute_list_begin()
	
	# fully connected
	rd.compute_list_bind_compute_pipeline(cl, _fc_pipeline)
	rd.compute_list_bind_uniform_set(cl, _input_set, 0)
	rd.compute_list_bind_uniform_set(cl, _layer_set, 1)
	rd.compute_list_bind_uniform_set(cl, _output_set, 2)
	rd.compute_list_dispatch(cl, groups, 1, 1)
	
	rd.compute_list_add_barrier(cl)
	
	# activation
	rd.compute_list_bind_compute_pipeline(cl, _sigmoid_pipeline)
	rd.compute_list_bind_uniform_set(cl, _output_set, 0)
	rd.compute_list_dispatch(cl, groups, 1, 1)
	
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	return rd.buffer_get_data(_output_buf).to_float32_array()
