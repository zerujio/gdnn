extends Control

@onready var gen_label: Label = $MarginContainer/VBoxContainer/Generation
@onready var gen_time_label: Label = $MarginContainer/VBoxContainer/Time
@onready var size_spinbox: SpinBox = $MarginContainer/VBoxContainer/Size/SpinBox
@onready var duration_spinbox: SpinBox = $MarginContainer/VBoxContainer/Duration/SpinBox


func set_gen(i: int) -> void:
	gen_label.text = "Generation %d" % i


func set_gen_time(t: float) -> void:
	gen_time_label.text = "%.2fs" % t
