extends Node2D

const Agent := preload("res://demo/pathfinding/agent.gd")
const AGENT_SCENE := preload("res://demo/pathfinding/agent.tscn")

@export var generation_duration := 15.0

var agents: Array[Agent]
var agent_loss: PackedFloat32Array
var generation := 0
var generation_time := 0.0

@onready var nn: NNMultiInstance = $NNMultiInstance
@onready var spawn_point: Node2D = $SpawnPoint
@onready var target: Node2D = $Target
@onready var ui := $CanvasLayer/UI


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
	
	ui.size_spinbox.value = agents.size()
	ui.duration_spinbox.value = generation_duration
	
	start_generation()


func _physics_process(delta: float) -> void:
	nn.submit_input()
	
	generation_time += delta
	ui.set_gen_time(generation_time)
	
	if generation_time > generation_duration:
		end_generation()
		start_generation()


func spawn_agent() -> Agent:
	var agent: Agent = AGENT_SCENE.instantiate()
	agent.nn = nn
	agent.nn_index = agents.size()
	agents.push_back(agent)
	add_child(agent)
	agent.process_physics_priority = process_physics_priority - 1
	return agent


func start_generation() -> void:
	for a in agents:
		a.global_position = spawn_point.global_position
	
	generation += 1
	generation_time = 0.0
	ui.set_gen(generation)


func end_generation() -> void:
	agent_loss.resize(agents.size())
	for i in range(agents.size()):
		agent_loss[i] = agents[i].global_position.distance_to(target.global_position)
	
	#todo: crossover


func _random_float_fill(data: PackedByteArray, range_min := -1.0, range_max := 1.0) -> void:
	var i := 0
	while i < data.size():
		data.encode_float(i, randf_range(range_min, range_max))
		i += 4


func _ui_duration_changed(value: float) -> void:
	generation_duration = value


func _ui_gen_size_changed(value: float) -> void:
	#todo
	pass
