extends Control

const Compute := preload("res://compute/compute.gd")

@onready var comp: Compute = $Compute

var input_tex: ImageTexture
var w_tex: ImageTexture
var b_tex: ImageTexture
var output_tex: ImageTexture


func _ready() -> void:
	var input := _init_tex("input_tex", comp.input_count, 1)
	$HBoxContainer/Input/TextureRect.texture = input_tex
	
	var w := _init_tex("w_tex", comp.input_count, comp.output_count)
	$HBoxContainer/Weight/TextureRect.texture = w_tex
	
	var b := _init_tex("b_tex", comp.output_count, 1)
	$HBoxContainer/Bias/TextureRect.texture = b_tex
	
	_init_tex("output_tex", comp.output_count, 1)
	$HBoxContainer/Output/TextureRect.texture = output_tex
	
	comp.update_weight(0, comp.instance_count, w)
	comp.update_bias(0, comp.instance_count, b)
	comp.submit_input(input)
	_queue_read_output()


func _init_tex(tex_property: StringName, width: int, height: int) -> PackedByteArray:
	var data := _random_array(width * height * comp.instance_count).to_byte_array()
	var img := Image.create_from_data(width, height * comp.instance_count, false, Image.FORMAT_RF, data)
	var tex := ImageTexture.create_from_image(img)
	set(tex_property, tex)
	return data


func _random_array(s: int) -> PackedFloat32Array:
	var data: PackedFloat32Array
	while data.size() < s:
		data.append(randf())
	return data


func _on_randomize_input_pressed() -> void:
	var input := _random_array(comp.instance_count * comp.input_count).to_byte_array()
	var img := Image.create_from_data(comp.input_count, comp.instance_count,
		false, Image.FORMAT_RF, input)
	input_tex.update(img)
	comp.submit_input(input)
	_queue_read_output()


func _on_randomize_weight_pressed() -> void:
	var w := _random_array(comp.instance_count * comp.input_count * comp.output_count).to_byte_array()
	var img := Image.create_from_data(comp.input_count, comp.output_count * comp.instance_count,
		false, Image.FORMAT_RF, w)
	w_tex.update(img)
	comp.update_weight(0, comp.instance_count, w)
	comp.submit_input()
	_queue_read_output()


func _on_randomize_bias_pressed() -> void: 
	var b := _random_array(comp.instance_count * comp.output_count).to_byte_array()
	var img := Image.create_from_data(comp.output_count, comp.instance_count,
		false, Image.FORMAT_RF, b)
	b_tex.update(img)
	comp.update_bias(0, comp.instance_count, b)
	comp.submit_input()
	_queue_read_output()


func _queue_read_output() -> void:
	get_tree().process_frame.connect(_read_output, CONNECT_ONE_SHOT)


func _read_output() -> void:
	var data := comp.sync_output()
	var img := Image.create_from_data(comp.output_count, comp.instance_count, 
		false, Image.FORMAT_RF, data)
	output_tex.update(img)
