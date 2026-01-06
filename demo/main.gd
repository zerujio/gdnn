extends Node2D

const Agent := preload("res://demo/pathfinding/agent.gd")
const AGENT_SCENE := preload("res://demo/pathfinding/agent.tscn")

var agents: Array[Agent]

@onready var nn: NNMultiInstance = $NNMultiInstance
@onready var spawn_point: Marker2D = $SpawnPoint


func _ready() -> void:
	assert(nn and nn.params and nn.params.layout)
	
	for i in range(nn.params.count):
		spawn_agent()
	
	var w: PackedByteArray
	var b: PackedByteArray
	var size := Vector2i(nn.params.layout.get_input_size(), 0)
	for i in range(nn.params.layout.get_layer_count()):
		size.y = nn.params.layout.get_layer_output_size(i)
		w.resize(size.x * size.y * agents.size() * 4)
		b.resize(size.y * agents.size() * 4)
		_random_float_fill(w)
		_random_float_fill(b)
		nn.params.update_layer_weights(i, w)
		nn.params.update_layer_bias(i, b)
		size.x = size.y


func _physics_process(_delta: float) -> void:
	nn.submit_input()


func _random_float_fill(data: PackedByteArray, range_min := -1.0, range_max := 1.0) -> void:
	var i := 0
	while i < data.size():
		data.encode_float(i, randf_range(range_min, range_max))
		i += 4


func spawn_agent() -> Agent:
	var agent: Agent = AGENT_SCENE.instantiate()
	agent.nn = nn
	agent.nn_index = agents.size()
	agents.push_back(agent)
	add_child(agent)
	agent.global_position = spawn_point.global_position
	agent.process_physics_priority = process_physics_priority - 1
	return agent
