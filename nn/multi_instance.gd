class_name NNMultiInstance
extends Node
## Contains multiple neural network instances (parameter sets) with the same layout.

enum LayerRID {
	IN_SET,
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
var capacity: int = 0:
	set = set_capacity

## Input storage buffer.
var _input_buf: RID

## RIDs of the bias buffer, weight buffer, weight uniform set, output buffer, 
## and output uniform set of each layer, in that order.
## e.g. : [b_buf_0, w_buf_0, w_set_0, out_buf_0, out_set_0, b_buf_1, w_buf_1, ..., out_set_n]
var _layer_rids: Array[RID]

## Cached input/output
var _input_cache: PackedByteArray
var _output_cache: PackedByteArray

## Neural network compute context. Warning: changing this will discard all stored data.
var ctx: NNContext:
	set(c):
		if ctx:
			assert(ctx._instances.count(self) == 1)
			ctx._instances.erase(self)
			if capacity > 0:
				_free_rids()
		ctx = c
		if ctx:
			assert(not ctx._instances.has(self))
			ctx._instances.append(self)
			if capacity > 0:
				_reallocate_buffers()


func _enter_tree() -> void:
	if ctx:
		_reallocate_buffers()


func _ready() -> void:
	if not ctx: 
		ctx = get_node(_context)


func _exit_tree() -> void:
	if ctx:
		_free_rids()


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
func set_input(instance_idx: int, data: PackedFloat32Array) -> void:
	assert(instance_idx >= -count and instance_idx < count)
	assert(data.size() % layout.get_input_size() == 0)
	const STRIDE := 4
	var offset := instance_idx * STRIDE
	for x in data:
		_input_cache.encode_float(offset, x)
		offset += STRIDE


## Retrieves the output data of one or more instances.
## [b]Note[/b]: accessing output data requires a CPU-GPU sync.
func get_output(instance_idx: int, instance_count := 1) -> PackedFloat32Array:
	assert(instance_idx >= 0)
	assert(instance_count > 0)
	assert(instance_idx + instance_count <= count)
	var stride := layout.get_output_size() * 4
	var begin := instance_idx * stride
	var end := begin + instance_count * stride
	return _output_cache.slice(begin, end).to_float32_array()


## Updates the weights at layer [param layer_idx] of one or more instances,
## starting at [param instance_idx].
func update_layer_weights(layer_idx: int, instance_idx: int, data: PackedByteArray) -> void: 
	var layer_size := layout.get_layer_size(layer_idx)
	var buf := _layer_rid(layer_idx, LayerRID.WEIGHT_BUFFER)
	_update_buffer(buf, instance_idx, layer_size.x * layer_size.y * 4, data)


## Updates the bias at layer [param layer_idx] of one or more instances, 
## starting at [param instance_idx].
func update_layer_bias(layer_idx: int, instance_idx: int, data: PackedByteArray) -> void:
	var buf := _layer_rid(layer_idx, LayerRID.BIAS_BUFFER)
	_update_buffer(buf, instance_idx, layout.get_layer_output_size(layer_idx) * 4, data)


func _submit_input() -> void:
	# upload input data
	_update_buffer(_input_buf, 0, layout.get_input_size() * 4, _input_cache)
	
	var layer_indices := range(layout.get_layer_count())
	
	# initialize output buffers to the bias values
	var rd := ctx.rd
	for i in layer_indices:
		var bias := _layer_rid(i, LayerRID.BIAS_BUFFER)
		var out := _layer_rid(i, LayerRID.OUT_BUFFER)
		rd.buffer_copy(bias, out, 0, 0, layout.get_layer_output_size(i) * 4 * count)
	
	# dispatch compute shaders
	var cl := rd.compute_list_begin()
	var input_log2 := layout.input_log2
	for i in layer_indices:
		var input_set := _layer_rid(i, LayerRID.IN_SET)
		var output_set := _layer_rid(i, LayerRID.OUT_SET)
		var output_log2 := layout.get_layer_output_log2(i)
		var output_count := 2 ** output_log2
		
		var weight_set := _layer_rid(i, LayerRID.WEIGHT_SET)
		var weight_count := 2 ** (input_log2 + output_log2)
		
		var pc: PackedByteArray
		pc.resize(16)
		pc.encode_u32(0, input_log2)
		pc.encode_u32(4, output_log2)
		
		# apply weights
		rd.compute_list_bind_compute_pipeline(cl, ctx._weight_pipeline)
		rd.compute_list_set_push_constant(cl, pc, pc.size())
		rd.compute_list_bind_uniform_set(cl, input_set, 0)
		rd.compute_list_bind_uniform_set(cl, weight_set, 1)
		rd.compute_list_bind_uniform_set(cl, output_set, 2)
		assert((weight_count * capacity) % NNContext.GROUP_SIZE == 0)
		var wg_count := ceili(float(count * weight_count) / NNContext.GROUP_SIZE)
		rd.compute_list_dispatch(cl, wg_count, 1, 1)
		
		# apply activation function
		var activation := ctx._activation_pipeline[layout.get_layer_activation(i)]
		if activation:
			rd.compute_list_add_barrier(cl)
			rd.compute_list_bind_compute_pipeline(cl, activation)
			rd.compute_list_bind_uniform_set(cl, output_set, 0)
			assert((output_count * capacity) % NNContext.GROUP_SIZE == 0)
			wg_count = ceili(float(count * output_count) / NNContext.GROUP_SIZE)
			rd.compute_list_dispatch(cl, wg_count, 1, 1)
		
		# outputs of this layer are inputs of the next
		rd.compute_list_add_barrier(cl)
		input_log2 = output_log2
	
	rd.compute_list_end()


func _sync_output():
	var buf := _input_buf if _layer_rids.is_empty() else _layer_rid(-1, LayerRID.OUT_BUFFER)
	_output_cache = ctx.rd.buffer_get_data(buf, 0, count * layout.get_output_size() * 4)


func _layout_changed() -> void:
	if ctx:
		_free_rids()
	_layer_rids.resize(layout.get_layer_count() * LayerRID.size() if layout else 0)
	if ctx:
		_reallocate_buffers()


func _layer_rid(layer_idx: int, type: LayerRID) -> RID:
	assert(layer_idx >= -layout.get_layer_count())
	assert(layer_idx < layout.get_layer_count())
	return _layer_rids[layer_idx * LayerRID.size() + type]


func _free_rids() -> void:
	ctx.rd.free_rid(_input_buf)
	for rid in _layer_rids:
		ctx.rd.free_rid(rid)
	_input_buf = RID()
	_layer_rids.fill(RID())


func _reallocate_buffers() -> void:
	const FLOAT_SIZE := 4
	var input_size := layout.get_input_size()
	_input_buf = _reallocate_buffer(_input_buf, FLOAT_SIZE * input_size)
	_input_cache.resize(FLOAT_SIZE * input_size * capacity)
	
	var input_buf := _input_buf
	for i in range(layout.get_layer_count()):
		var output_size := layout.get_output_size()
		var j := LayerRID.size() * i
		
		# input
		_layer_rids[j + LayerRID.IN_SET] = ctx.create_input_uniform_set(input_buf)
		
		# output
		var ob := j + LayerRID.OUT_BUFFER
		var output_buf := _reallocate_buffer(_layer_rids[ob], FLOAT_SIZE * output_size)
		_layer_rids[ob] = output_buf
		_layer_rids[j + LayerRID.OUT_SET] = ctx.create_output_uniform_set(output_buf)
		
		# weight
		var wb := j + LayerRID.WEIGHT_BUFFER
		_layer_rids[wb] = _reallocate_buffer(_layer_rids[wb], FLOAT_SIZE * input_size * output_size)
		_layer_rids[j + LayerRID.WEIGHT_SET] = ctx.create_weight_uniform_set(_layer_rids[wb])
		
		# bias
		var bb := j + LayerRID.BIAS_BUFFER
		_layer_rids[bb] = _reallocate_buffer(_layer_rids[bb], FLOAT_SIZE * output_size)
		
		input_size = output_size
		input_buf = output_buf


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
	assert(idx * stride + data.size() <= count * stride)
	ctx.rd.buffer_update(buf, idx * stride, data.size(), data)
