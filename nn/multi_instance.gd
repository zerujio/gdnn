class_name NNMultiInstance
extends Node
## Contains multiple neural network instances (parameter sets) with the same layout.

enum LayerRID { 
	OUT_BUFFER,
	OUT_SET,
	WEIGHT_BUFFER, 
	WEIGHT_SET,
	BIAS_BUFFER
}

## Neural network compute context. 
@export_node_path("NNContext") var _context: NodePath = ".."

## Neural network layout.
##
## [b]Warning:[/b] changing this resource will discard all instance data.
@export var layout: NNLayout:
	set(l):
		if layout: 
			layout.changed.disconnect(_layout_changed)
		layout = l
		if layout:
			layout.changed.connect(_layout_changed)
		_layout_changed()

## Number of active instances. Each instance has the same layout, but different parameters.
@export var count: int = 0:
	set = set_count

## Number of instances that would fit in the currently allocated memory.
@export var capacity: int = 0:
	set = set_capacity

## Input storage buffer.
var _input_buf: RID

## Input unifom set.
var _input_set: RID

## RIDs of the bias buffer, weight buffer, weight uniform set, output buffer, 
## and output uniform set of each layer, in that order.
## e.g. : [b_buf_0, w_buf_0, w_set_0, out_buf_0, out_set_0, b_buf_1, w_buf_1, ..., out_set_n]
var _layer_rids: Array[RID]

## Neural network compute context. Warning: changing this discards all stored information.
@onready var ctx: NNContext = get_node(_context):
	set(c):
		_free_rids()
		ctx = c
		_reallocate_buffers()


func set_count(new_count: int) -> void:
	assert(new_count >= 0)
	new_count = maxi(0, new_count)
	if new_count > capacity:
		var new_cap := maxi(1, capacity)
		while new_cap < new_count:
			new_cap *= 2
		capacity = new_cap
	count = new_count


func set_capacity(new_cap: int) -> void:
	assert(new_cap >= 0)
	new_cap = maxi(0, new_cap)
	
	if new_cap == capacity:
		return
	
	capacity = new_cap
	
	if count > capacity:
		count = capacity
	
	_reallocate_buffers()


## Updates the input data for one or more instances starting at [param instance_idx].
## Number of instances updated depends on size of [param data].
func set_input(instance_idx: int, data: PackedByteArray) -> void:
	_update_buffer(_input_buf, instance_idx, layout.input_size * 4, data)


## Retrieves the output data of one or more instances.
## [b]Note[/b]: accessing output data requires a CPU-GPU sync.
func get_output(instance_idx: int, instance_count := 1) -> void:
	assert(instance_idx >= 0)
	assert(instance_count > 0)
	assert(instance_idx + instance_count <= count)
	var buf := _input_buf if _layer_rids.is_empty() else _layer_rid(-1, LayerRID.OUT_BUFFER)
	var stride := layout.get_output_size() * 4
	return ctx.rd.buffer_get_data(buf, instance_idx * stride, instance_count * stride)


## Updates the weights at layer [param layer_idx] of one or more instances,
## starting at [param instance_idx].
func update_layer_weights(layer_idx: int, instance_idx: int, data: PackedByteArray) -> void: 
	var layer_size := layout.get_layer_size(layer_idx)
	var buf := _layer_rid(layer_idx, LayerRID.WEIGHT_BUFFER)
	_update_buffer(buf, instance_idx, layer_size.x * layer_size.y, data)


## Updates the bias at layer [param layer_idx] of one or more instances, 
## starting at [param instance_idx].
func update_layer_bias(layer_idx: int, instance_idx: int, data: PackedByteArray) -> void:
	var buf := _layer_rid(layer_idx, LayerRID.BIAS_BUFFER)
	_update_buffer(buf, instance_idx, layout.get_layer_output_size(layer_idx) * 4, data)


func _layout_changed() -> void:
	_free_rids()
	_layer_rids.resize(layout.get_layer_count() * 5 if layout else 0)
	_reallocate_buffers()


func _layer_rid(layer_idx: int, type: LayerRID) -> RID:
	assert(layer_idx >= -layout.get_layer_count())
	assert(layer_idx < layout.get_layer_count())
	return _layer_rids[layer_idx * 5 + type]


func _free_rids() -> void:
	if not ctx:
		return
	
	for rid in [_input_buf, _input_set] + _layer_rids:
		ctx.rd.free_rid(rid)
	_input_buf = RID()
	_input_set = RID()
	_layer_rids.fill(RID())


func _reallocate_buffers() -> void:
	if not ctx:
		return
	
	const FLOAT_SIZE := 4
	_input_buf = _reallocate_buffer(_input_buf, FLOAT_SIZE * layout.input_size)
	_input_set = ctx.create_io_uniform_set(_input_buf)
	
	var input_size := layout.input_size
	for i in range(layout.get_layer_count()):
		var output_size := layout.get_output_size()
		var j := 5 * i
		
		# output
		var ob := j + LayerRID.OUT_BUFFER
		var os := j + LayerRID.OUT_SET
		_layer_rids[ob] = _reallocate_buffer(_layer_rids[ob], FLOAT_SIZE * output_size)
		_layer_rids[os] = ctx.create_io_uniform_set(_layer_rids[os])
		
		# weight
		var wb := j + LayerRID.WEIGHT_BUFFER
		var ws := j + LayerRID.WEIGHT_SET
		_layer_rids[wb] = _reallocate_buffer(_layer_rids[wb], FLOAT_SIZE * input_size * output_size)
		_layer_rids[ws] = ctx.create_io_uniform_set(_layer_rids[ws])
		
		# bias
		var bb := j + LayerRID.BIAS_BUFFER
		_layer_rids[bb] = _reallocate_buffer(_layer_rids[bb], FLOAT_SIZE * output_size)
		
		input_size = output_size


func _reallocate_buffer(old_buf: RID, element_size: int) -> RID:
	var new_buf := ctx.rd.storage_buffer_create(capacity * element_size)
	if old_buf:
		ctx.rd.buffer_copy(old_buf, new_buf, 0, 0, count * element_size)
	return new_buf


func _update_buffer(buf: RID, idx: int, stride: int, data: PackedByteArray) -> void:
	assert(buf)
	assert(idx >= 0)
	assert(stride >= 0)
	assert(data.size() > 0)
	assert(idx * stride + data.size() < count * stride)
	ctx.rd.buffer_update(buf, idx * stride, data.size(), data)
