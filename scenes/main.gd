extends Node2D

const Agent := preload("res://scenes/agent.gd")
const AGENT_SCENE := preload("res://scenes/agent.tscn")

## Number of agents to spawn.
@export_range(1, 2048) var agent_count := 32
@export var nn_params: NNRandomParams

var agents: Array[Agent]

@onready var nn_ctx: NNContext = $NNContext
@onready var nn: NNMultiInstance = $NNContext/NNMultiInstance
@onready var spawn_point: Marker2D = $SpawnPoint


func _ready() -> void:
	nn.count = agent_count
	for i in range(agent_count):
		spawn_agent()
	
	var w: PackedByteArray
	var b: PackedByteArray
	var input_size := nn.layout.get_input_size()
	for i in range(nn.layout.get_layer_count()):
		var output_size := nn.layout.get_layer_output_size(i)
		w.resize(input_size * output_size * agents.size() * 4)
		b.resize(output_size * agents.size() * 4)
		var w_offset := 0
		var b_offset := 0
		for a in agents:
			nn_params.copy_to_buffer(i, NNParams.Type.WEIGHT, w, w_offset)
			nn_params.copy_to_buffer(i, NNParams.Type.BIAS, b, b_offset)
			w_offset += (input_size + output_size) * 4
			b_offset += output_size * 4
		nn.update_layer_weights(i, 0, w)
		nn.update_layer_bias(i, 0, b)
		input_size = output_size
	
	
	# call sync at the start of each frame, skipping the first one
	get_tree().physics_frame.connect(func():
		get_tree().physics_frame.connect(_pre_physics_frame),
		CONNECT_ONE_SHOT)


func _physics_process(_delta: float) -> void:
	nn_ctx.submit_input()


func _pre_physics_frame() -> void:
	nn_ctx.sync_output()


func spawn_agent() -> Agent:
	var agent: Agent = AGENT_SCENE.instantiate()
	agent.nn = nn
	agent.nn_index = agents.size()
	agents.push_back(agent)
	add_child(agent)
	agent.global_position = spawn_point.global_position
	return agent
