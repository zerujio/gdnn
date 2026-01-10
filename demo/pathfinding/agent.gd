extends CharacterBody2D

@export var speed := 48.0
@export var nn: NNMultiInstance
@export var nn_index: int
@export var input_scale := Vector2.ONE

var _nn_input: PackedFloat32Array

@onready var raycasts: Array[RayCast2D] = [
	$RayCast2D,
	$RayCast2D2,
	$RayCast2D3,
	$RayCast2D4,
	$RayCast2D5,
	$RayCast2D6
]


func _ready() -> void:
	_nn_input.resize(2)
	$Label.text = str(nn_index)


func _physics_process(_delta: float) -> void:
	if not nn:
		return
	
	# don't need this data in the basic use case
	#var i := 0
	#for rc in raycasts:
		#var d := rc.get_collision_point().distance_to(global_position) if rc.is_colliding() else 0.0
		#_nn_input[i] = d
		#i += 1
	
	var p := global_position
	_nn_input[0] = p.x * input_scale.x
	_nn_input[1] = p.y * input_scale.y
	
	nn.set_input(nn_index, _nn_input)
	
	var output := nn.get_output(nn_index)
	if output.is_empty():
		return
	
	var dir := (Vector2(output[0], output[1]).normalized())
	velocity = dir * speed
	
	move_and_slide()
