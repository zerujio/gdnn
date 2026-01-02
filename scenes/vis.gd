extends Control

@onready var nn: NNProcessor = $NeuralNetwork

var input_tex: ImageTexture
var w_tex: ImageTexture
var b_tex: ImageTexture
var output_tex: ImageTexture


func _ready() -> void:
	var input := _init_tex("input_tex", nn.input_count, 1)
	$HBoxContainer/Input/TextureRect.texture = input_tex
	
	var w := _init_tex("w_tex", nn.input_count, nn.output_count)
	$HBoxContainer/Weight/TextureRect.texture = w_tex
	
	var b := _init_tex("b_tex", nn.output_count, 1)
	$HBoxContainer/Bias/TextureRect.texture = b_tex
	
	_init_tex("output_tex", nn.output_count, 1)
	$HBoxContainer/Output/TextureRect.texture = output_tex
	
	nn.update_weight(0, nn.instance_count, w)
	nn.update_bias(0, nn.instance_count, b)
	nn.submit_input(input)
	_queue_read_output()


func _init_tex(tex_property: StringName, width: int, height: int) -> PackedByteArray:
	var data := _random_array(width * height * nn.instance_count).to_byte_array()
	var img := Image.create_from_data(width, height * nn.instance_count, false, Image.FORMAT_RF, data)
	var tex := ImageTexture.create_from_image(img)
	set(tex_property, tex)
	return data


func _random_array(s: int) -> PackedFloat32Array:
	var data: PackedFloat32Array
	while data.size() < s:
		data.append(randf())
	return data


func _on_randomize_input_pressed() -> void:
	var input := _random_array(nn.instance_count * nn.input_count).to_byte_array()
	var img := Image.create_from_data(nn.input_count, nn.instance_count,
		false, Image.FORMAT_RF, input)
	input_tex.update(img)
	nn.submit_input(input)
	_queue_read_output()


func _on_randomize_weight_pressed() -> void:
	var w := _random_array(nn.instance_count * nn.input_count * nn.output_count).to_byte_array()
	var img := Image.create_from_data(nn.input_count, nn.output_count * nn.instance_count,
		false, Image.FORMAT_RF, w)
	w_tex.update(img)
	nn.update_weight(0, nn.instance_count, w)
	nn.submit_input()
	_queue_read_output()


func _on_randomize_bias_pressed() -> void: 
	var b := _random_array(nn.instance_count * nn.output_count).to_byte_array()
	var img := Image.create_from_data(nn.output_count, nn.instance_count,
		false, Image.FORMAT_RF, b)
	b_tex.update(img)
	nn.update_bias(0, nn.instance_count, b)
	nn.submit_input()
	_queue_read_output()


func _queue_read_output() -> void:
	get_tree().process_frame.connect(_read_output, CONNECT_ONE_SHOT)


func _read_output() -> void:
	var data := nn.sync_output()
	var img := Image.create_from_data(nn.output_count, nn.instance_count, 
		false, Image.FORMAT_RF, data)
	output_tex.update(img)
