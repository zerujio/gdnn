extends Control

const Compute := preload("res://compute/compute.gd")

@onready var label: Label = $Label
@onready var comp: Compute = $Compute


func _ready() -> void:
	var input := _random_array(comp.input_size)
	comp.set_input(input)
	label.text += "input: %s\n" % input
	
	var weight := _random_array(comp.input_size * comp.output_size)
	var bias := _random_array(comp.output_size)
	comp.set_weight_and_bias(weight, bias)
	label.text += "weights: %s\nbiases: %s\n" % [weight, bias]
	
	var output := comp.dispatch()
	label.text += "output: %s\n" % output


func _random_array(s: int) -> PackedFloat32Array:
	var data: PackedFloat32Array
	while data.size() < s:
		data.append(randf())
	return data
