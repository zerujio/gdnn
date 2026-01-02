class_name NNInstance
extends Node

@export var processor: NNProcessor:
	set = set_processor

var instance_idx: int = -1


func _enter_tree() -> void:
	if processor:
		instance_idx = processor.add_instance()


func _exit_tree() -> void:
	if processor and instance_idx > -1:
		processor.remove_instance(instance_idx)


func set_processor(p: NNProcessor) -> void:
	if processor and instance_idx > -1:
		processor.remove_instance(instance_idx)
		instance_idx = -1
	processor = p
	if processor:
		instance_idx = processor.add_instance()


func set_params(data: PackedFloat32Array) -> void:
	processor.set_instance_params(instance_idx, data)


func set_input(data: PackedFloat32Array) -> void:
	processor.set_instance_input(instance_idx, data)


func get_output() -> PackedFloat32Array:
	return processor.get_instance_output(instance_idx)
