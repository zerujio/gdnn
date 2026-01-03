extends Node2D

const Agent := preload("res://scenes/agent.gd")
const AGENT_SCENE := preload("res://scenes/agent.tscn")

## Number of agents to spawn.
@export_range(1, 2048) var agent_count := 32

var agents: Array[Agent]

@onready var nn_proc: NNProcessor = $NNProcessor
@onready var spawn_point: Marker2D = $SpawnPoint


func _ready() -> void:
	nn_proc.context.set_capacity(agent_count)
	
	# neural network params
	var params: PackedFloat32Array
	params.resize(nn_proc.get_weight_count() + nn_proc.get_output_count())
	
	for i in range(agent_count):
		var agent := spawn_agent()
		# randomize initial params
		for j in range(params.size()):
			params[j] = randf_range(-1.0, 1.0)
		agent.nn.set_params(params)


func spawn_agent() -> Agent:
	var agent: Agent = AGENT_SCENE.instantiate()
	var nn: NNInstance = agent.get_node("NNInstance")
	nn.processor = nn_proc
	agents.push_back(agent)
	add_child(agent)
	assert(agent.nn.instance_idx == agents.size() - 1)
	agent.global_position = spawn_point.global_position
	return agent
