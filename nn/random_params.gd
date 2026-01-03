class_name NNRandomParams
extends NNParams
## Generates random neural network parameters with a uniform distribution.
## The parameter values are generated using [method randf_range] each time 
## [method copy_to_buffer] is called. This means that calling said function 
## with the same arguments will not return the same values each time (unless 
## the random seed is re-set using [method seed]).

## Random seed.
@export var range_min := 0.0
@export var range_max := 1.0


func copy_to_buffer(layer_idx: int, param_type: Type, buffer: PackedByteArray, buffer_offset := 0) -> void:
	if not layout:
		return
	
	assert(layer_idx >= 0 and layer_idx < layout.get_layer_count(),
		"layer index out of range: %d (layer count is %d)" % [layer_idx, layout.get_layer_count()])
	
	var count := layout.get_layer_output_size(layer_idx)
	if param_type == NNParams.Type.WEIGHT:
		count *= layout.get_layer_input_size(layer_idx)
	
	var size := count * 4
	assert(buffer.size() - buffer_offset >= size)
	
	for i in range(buffer_offset, buffer_offset + size, 4):
		buffer.encode_float(i, randf_range(range_min, range_max))
