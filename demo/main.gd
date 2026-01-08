extends Node2D

const UI := preload("res://demo/pathfinding/ui/ui.gd")
const Agent := preload("res://demo/pathfinding/agent.gd")
const AGENT_SCENE := preload("res://demo/pathfinding/agent.tscn")

var agents: Array[Agent]
var generation := 0

@onready var timer: Timer = $Timer
@onready var nn: NNMultiInstance = $NNMultiInstance
@onready var spawn_point: Node2D = $SpawnPoint
@onready var target: Node2D = $Target
@onready var ui: UI = $CanvasLayer/UI


func _ready() -> void:
	assert(nn and nn.params and nn.params.layout)
	
	nn.params.randomize_params()
	
	for i in range(nn.params.count):
		spawn_agent()
	
	ui.size_spinbox.value = agents.size()
	ui.gen_time_spinbox.value = timer.wait_time
	timer.start()
	
	start_generation()


func _physics_process(_delta: float) -> void:
	nn.submit_input()
	ui.set_time_left(timer.time_left)


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
	
	ui.set_gen(generation)
	
	timer.start(ui.gen_time_spinbox.value)
	ui.set_time_left(timer.time_left)
	
	prints("generation", generation)
	var s := "weight:"
	for w in nn.params._weight_buffer(0).get_data().to_float32_array():
		s += " %.1f" % w
	print(s)
	s = "bias:"
	for b in nn.params._weight_buffer(0).get_data().to_float32_array():
		s += " %.1f" % b
	print(s)


func crossover(selection_ratio: float) -> void:
	var loss := calculate_agents_loss()
	
	var indices := range(agents.size())
	indices.sort_custom(func(i: int, j: int): return loss[i] < loss[j])
	indices.resize(ceili(indices.size() * selection_ratio))
	assert(not indices.is_empty())
	
	var pairs: PackedInt32Array
	pairs.resize(agents.size() * 2)
	for i in range(0, pairs.size(), 2):
		pairs[0] = indices.pick_random()
		pairs[1] = indices.pick_random()
	
	nn.params = nn.params.intermediate_crossover(pairs)


func calculate_agents_loss() -> PackedFloat32Array:
	var loss: PackedFloat32Array
	loss.resize(agents.size())
	for i in range(agents.size()):
		loss[i] = agents[i].global_position.distance_squared_to(target.global_position)
	return loss


func on_timeout() -> void:
	crossover(0.25)
	nn.params.randomize_params(-0.05, 0.05)
	generation += 1
	start_generation()
