class_name NNContext
extends Node
## GPU context for neural network inference.

const WEIGHT_SHADER_FILE: RDShaderFile = preload("res://nn/compute/weight.glsl")
const ACTIVATION_SHADER_FILE: RDShaderFile = preload("res://nn/compute/activation.glsl")
const INPUT_EXP_MAX := 6
const GROUP_SIZE := 2 ** INPUT_EXP_MAX

## Rendering device used for compute.
var rd := RenderingServer.create_local_rendering_device()

## Neural network instances managed by this context.
var _instances: Array[NNMultiInstance]

## Multiplies input by layer weights.
var _weight_shader: RID = rd.shader_create_from_spirv(WEIGHT_SHADER_FILE.get_spirv(), "weight")
var _weight_pipeline: RID = rd.compute_pipeline_create(_weight_shader)

## Applies activation functions.
var _activation_shader: RID = rd.shader_create_from_spirv(ACTIVATION_SHADER_FILE.get_spirv(), "activation")
var _activation_pipeline: Array[RID]


func _init() -> void:
	# activation functions
	_activation_pipeline.resize(NNLayout.Activation.size())
	var spec_const := RDPipelineSpecializationConstant.new()
	for i in range(1, NNLayout.Activation.size()):
		spec_const.value = i - 1
		_activation_pipeline[i] = rd.compute_pipeline_create(_activation_shader, [spec_const])


## Get the list of instances associated with this context.
func get_instances() -> Array[NNMultiInstance]:
	return _instances.duplicate()


## Gathers input from all instances associated with this context and dispatches
## the compute operations.
func submit_input() -> void:
	for instance in _instances:
		instance._submit_input()
	rd.submit()


## Waits until all previously dispatched compute operations are finished, making
## its output available.
func sync_output() -> void:
	rd.sync()
	for instance in _instances:
		instance._sync_output()


func create_weight_uniform_set(buffer: RID) -> RID:
	return rd.uniform_set_create([_storage_buffer_uniform(buffer, 0)], _weight_shader, 0)


func create_input_uniform_set(buffer: RID) -> RID: 
	return rd.uniform_set_create([_storage_buffer_uniform(buffer, 0)], _weight_shader, 1)


func create_output_uniform_set(buffer: RID) -> RID:
	return rd.uniform_set_create([_storage_buffer_uniform(buffer, 0)], _weight_shader, 2)


func _storage_buffer_uniform(buffer: RID, binding: int) -> RDUniform:
	var uniform := RDUniform.new()
	uniform.binding = binding
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.add_id(buffer)
	return uniform
