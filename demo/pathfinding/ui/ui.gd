extends Control

@onready var gen_label: Label = $Status/Generation
@onready var gen_time_label: Label = $Status/Time
@onready var gen_time_spinbox: SpinBox = $Settings/MarginContainer/VBoxContainer/Duration/SpinBox
@onready var size_spinbox: SpinBox = $Settings/MarginContainer/VBoxContainer/Size/SpinBox


func set_gen(i: int) -> void:
	gen_label.text = "Generation %d" % i


func set_time_left(t: float) -> void:
	gen_time_label.text = "%.2fs" % t
