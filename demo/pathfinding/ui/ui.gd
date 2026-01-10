extends Control

@onready var gen_label: Label = $Status/Generation
@onready var gen_time_label: Label = $Status/Time
@onready var gen_time_spinbox: SpinBox = $Settings/MarginContainer/VBoxContainer/Duration/SpinBox
@onready var size_spinbox: SpinBox = $Settings/MarginContainer/VBoxContainer/Size/SpinBox
@onready var description_label: Label = $Settings/MarginContainer/VBoxContainer/Description


func set_gen(i: int) -> void:
	gen_label.text = "Generation %d" % i


func set_time_left(t: float) -> void:
	gen_time_label.text = "%.2fs" % t


func set_nn_layout(layout: NNLayout) -> void:
	description_label.text = "%d inputs, %d outputs.\n%d layers." % [
		layout.get_input_size(),
		layout.get_output_size(),
		layout.get_layer_count(),
	]
