extends CharacterBody2D

@export var nn_processor: NNProcessor
@export var speed := 64.0

@onready var nn: NNInstance = $NNInstance
@onready var raycasts: Array[RayCast2D] = [
	$RayCast2D,
	$RayCast2D2,
	$RayCast2D3,
	$RayCast2D4,
	$RayCast2D5,
	$RayCast2D6
]


func _ready() -> void:
	nn.processor = nn_processor


func _physics_process(_delta: float) -> void:
	var input: PackedFloat32Array
	
	for rc in raycasts:
		input.push_back(rc.get_collision_point().distance_to(global_position) if rc.is_colliding() else 0.0)
	
	var rv := get_real_velocity()
	input.push_back(rv.x)
	input.push_back(rv.y)
	
	nn.set_input(input)
	
	var output := nn.get_output()
	if output.is_empty():
		return
	
	var dir := Vector2(output[0], output[1]).normalized()
	velocity = dir * speed
	
	move_and_slide()
