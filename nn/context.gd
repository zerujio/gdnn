class_name NNContext
extends Node
## Compute context for neural networks.
## 
## [b]Note:[b/] this is a GDScript draft. Ideally, this would be implemented using GDExtension.

enum UnaryOp { SIGMOID, RELU }
enum TernaryOp { LERP }

const FLOAT_SIZE := 4
const WORK_GROUP_SIZE_LOG2 := 6
const WORK_GROUP_SIZE := 2 ** WORK_GROUP_SIZE_LOG2

const WEIGHT_SRC: RDShaderFile = preload("res://nn/compute/weight.glsl")
const UNARY_SRC: RDShaderFile = preload("res://nn/compute/activation.glsl")
const TERNARY_SRC: RDShaderFile = preload("res://nn/compute/ternary.glsl")

## If enabled, the runtime will dispatch compute workloads during [method _physics_process].
## Otherwise, it will dispatch them during [method _process].
var use_physics_process := true

## RenderingDevice used for compute.
var rd := RenderingServer.create_local_rendering_device()
var _submitted := false

var _weight_shader: RID = rd.shader_create_from_spirv(WEIGHT_SRC.get_spirv(), "weight")
var _weight_pipeline: RID = rd.compute_pipeline_create(_weight_shader)

var _unary_shader: RID = rd.shader_create_from_spirv(UNARY_SRC.get_spirv(), "unary")
var _unary_pipeline: Array[RID]

var _ternary_shader: RID = rd.shader_create_from_spirv(UNARY_SRC.get_spirv(), "ternary")
var _ternary_pipeline: RID


func _init() -> void:
	# higher priority so it will process after other nodes
	process_priority = 1
	process_physics_priority = 1
	
	var spec_const := RDPipelineSpecializationConstant.new()
	
	_unary_pipeline.resize(NNLayout.Activation.size())
	for i in range(1, NNLayout.Activation.size()):
		spec_const.value = i - 1
		_unary_pipeline[i] = rd.compute_pipeline_create(_unary_shader, [spec_const])
	
	spec_const.value = 0
	_ternary_pipeline = rd.compute_pipeline_create(_ternary_shader, [spec_const])


func _ready() -> void:
	set_physics_process(use_physics_process)
	set_process(not use_physics_process)
	
	if use_physics_process:
		get_tree().physics_frame.connect(_sync)
	else:
		get_tree().process_frame.connect(_sync)


func _process(_delta: float) -> void:
	_submit.call_deferred()


func _physics_process(_delta: float) -> void:
	_submit.call_deferred()


func _submit() -> void:
	assert(not _submitted)
	rd.submit()
	_submitted = true


func _sync() -> void:
	if _submitted:
		rd.sync()
		_submitted = false


## Allocates a an array buffer of floats.
func create_array_buffer(size: int, data := PackedByteArray()) -> ArrayBuffer:
	var buf := ArrayBuffer.new(self)
	buf.allocate(size, data)
	return buf


func create_vector_buffer(count: int, vector_size: int, data := PackedByteArray()) -> ArrayBuffer:
	var buf := ArrayBuffer.new(self)
	buf.allocate_vector(count, vector_size, data)
	return buf


func create_matrix_buffer(count: int, matrix_size: Vector2i, data := PackedByteArray()) -> ArrayBuffer:
	var buf := ArrayBuffer.new(self)
	buf.allocate_matrix(count, matrix_size, data)
	return buf


## Performs matrix-vector multiplication between [param in_matrix] and [param in_vector],
## and adds the result to [param out_vector]. 
## The Z dimension of the matrix argument must match the Y dimension of vector arguments.
func matrix_vector_multiply_add(in_matrix: ArrayBuffer, matrix_size_log2: Vector2i, 
		in_vector: ArrayBuffer, out_vector: ArrayBuffer) -> void:
	var size := Vector2i(2 ** matrix_size_log2.x, 2 ** matrix_size_log2.y)
	assert(size.x >= 0 and size.y >= 0)
	assert(in_matrix.size % (size.x * size.y) == 0)
	assert(in_vector.size % size.x == 0)
	assert(out_vector.size % size.y == 0)
	
	var pc: PackedByteArray
	pc.resize(16)
	pc.encode_u32(0, matrix_size_log2.x)
	pc.encode_u32(4, matrix_size_log2.y)
	
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, _weight_pipeline)
	rd.compute_list_set_push_constant(cl, pc, pc.size())
	rd.compute_list_bind_uniform_set(cl, in_matrix._read_uniform_set, 0)
	rd.compute_list_bind_uniform_set(cl, in_vector._read_uniform_set, 1)
	rd.compute_list_bind_uniform_set(cl, out_vector._write_uniform_set, 2)
	rd.compute_list_dispatch(cl, ceili(in_matrix.size / float(WORK_GROUP_SIZE)), 1, 1)
	rd.compute_list_end()


## Performs element-wise interpolation (lerp) between [param x] and [param y] by factor [param a],
## and writes to [param result].
func interpolate(x: ArrayBuffer, y: ArrayBuffer, a: ArrayBuffer, result: ArrayBuffer) -> void:
	var size := result.size
	assert(x.size == size)
	assert(y.size == size)
	assert(a.size == size)
	
	var cl := rd.compute_list_begin() 
	rd.compute_list_bind_compute_pipeline(cl, _ternary_pipeline)
	rd.compute_list_bind_uniform_set(cl, x._read_uniform_set, 0)
	rd.compute_list_bind_uniform_set(cl, y._read_uniform_set, 1)
	rd.compute_list_bind_uniform_set(cl, a._read_uniform_set, 2)
	rd.compute_list_bind_uniform_set(cl, result._write_uniform_set, 3)
	rd.compute_list_dispatch(cl, ceil(float(size) / WORK_GROUP_SIZE), 1, 1)
	rd.compute_list_end()


func sigmoid(src: ArrayBuffer, dst: ArrayBuffer) -> void:
	_unary(src, dst, 0)


func relu(src: ArrayBuffer, dst: ArrayBuffer) -> void:
	_unary(src, dst, 1)


func _unary(src: ArrayBuffer, dst: ArrayBuffer, idx: int) -> void:
	assert(src.size == dst.size)
	var cl := rd.compute_list_begin() 
	rd.compute_list_bind_compute_pipeline(cl, _unary_pipeline[idx])
	rd.compute_list_bind_uniform_set(cl, src._read_uniform_set, 0)
	rd.compute_list_bind_uniform_set(cl, dst._read_uniform_set, 1)
	rd.compute_list_dispatch(cl, ceil(float(src.size) / WORK_GROUP_SIZE), 1, 1)
	rd.compute_list_end()


class ArrayBuffer:
	var size: int:
		set(s): resize(s)
		get: return _size
	var _size: int
	var _ctx: NNContext
	var _buffer: RID
	var _read_uniform_set: RID
	var _write_uniform_set: RID
	
	
	func _init(ctx: NNContext) -> void:
		assert(ctx)
		_ctx = ctx
	
	
	func _notification(what: int) -> void:
		match what:
			NOTIFICATION_PREDELETE:
				_ctx.rd.free_rid(_buffer)
	
	
	func size_bytes() -> int:
		return size * FLOAT_SIZE
	
	
	## Allocates a new buffer, discarding any existing data.
	func allocate(new_size: int, data := PackedByteArray()) -> void:
		assert(data.size() % FLOAT_SIZE == 0)
		if _buffer:
			_ctx.rd.free_rid(_buffer)
		_buffer = _ctx.rd.storage_buffer_create(new_size * FLOAT_SIZE, data) if new_size > 0 else RID()
		_create_uniform_sets()
		_size = new_size
	
	
	func allocate_vector(new_count: int, vector_size, data := PackedByteArray()) -> void:
		assert(data.size() % vector_size == 0)
		allocate(new_count * vector_size, data)
	
	
	func allocate_matrix(new_count: int, matrix_size: Vector2i, data := PackedByteArray()) -> void:
		assert(data.size() % (matrix_size.x * matrix_size.y) == 0)
		allocate_vector(new_count * matrix_size.y, matrix_size.x, data)
	
	
	## Allocates a new buffer, copying over any existing data
	func resize(new_size: int) -> void:
		if new_size == size:
			return
		
		var old_buffer := _buffer
		_buffer = _ctx.rd.storage_buffer_create(new_size * FLOAT_SIZE) if new_size > 0 else RID()
		_create_uniform_sets()
		
		if old_buffer:
			if _buffer:
				_ctx.rd.buffer_copy(old_buffer, _buffer, 0, 0, mini(new_size, size) * FLOAT_SIZE)
			_ctx.rd.free_rid(old_buffer)
		
		_size = new_size
	
	
	func resize_vector(new_count: int, vector_size) -> void:
		resize(new_count * vector_size)
	
	
	func resize_matrix(new_count: int, matrix_size: Vector2i) -> void:
		resize_vector(new_count * matrix_size.y, matrix_size.x)
	
	
	func update(offset: int, data: PackedByteArray) -> Error:
		assert(_buffer)
		assert(data.size() % FLOAT_SIZE == 0)
		assert(data.size() <= (size - offset) * FLOAT_SIZE)
		return _ctx.rd.buffer_update(_buffer, offset * FLOAT_SIZE, data.size(), data)
	
	
	func update_vector(offset: int, vector_size: int, data: PackedByteArray) -> Error:
		assert(vector_size > 0)
		assert(data.size() % vector_size == 0)
		return _ctx.rd.buffer_update(_buffer, offset * vector_size * FLOAT_SIZE, data.size(), data)
	
	
	func update_matrix(offset: int, matrix_size: Vector2i, data: PackedByteArray) -> Error:
		assert(matrix_size.x > 0 and matrix_size.y > 0)
		assert(data.size() % (matrix_size.x * matrix_size.y) == 0)
		return _ctx.rd.buffer_update(_buffer, offset * matrix_size.x * matrix_size.y, data.size(), data)
	
	
	func clear(offset: int, count: int) -> Error:
		assert(offset >= 0)
		assert(count > 0)
		return _ctx.rd.buffer_clear(_buffer, offset * FLOAT_SIZE, count * FLOAT_SIZE)
	
	
	func clear_vector(offset: int, count: int, vector_size: int) -> Error:
		assert(vector_size > 0)
		return clear(offset * vector_size, count * vector_size)
	
	
	func clear_matrix(offset: int, count: int, matrix_size: Vector2i) -> Error:
		assert(matrix_size.y > 0)
		return clear_vector(offset * matrix_size.y, count * matrix_size.y, matrix_size.x)
	
	
	static func copy(src: ArrayBuffer, dst: ArrayBuffer, src_offset := 0, dst_offset := 0, count := 1) -> Error:
		assert(src and dst)
		assert(src._ctx == dst._ctx)
		return src._ctx.rd.buffer_copy(src._buffer, dst._buffer, src_offset * FLOAT_SIZE, dst_offset * FLOAT_SIZE, count * FLOAT_SIZE)
	
	
	static func copy_vector(src: ArrayBuffer, dst: ArrayBuffer,
			src_offset: int, dst_offset: int,
			count: int, vector_size: int) -> Error:
		assert(vector_size > 0)
		return copy(src, dst, src_offset * vector_size, dst_offset * vector_size, count * vector_size)
	
	
	static func copy_matrix(src: ArrayBuffer, dst: ArrayBuffer,
			src_offset: int, dst_offset: int,
			count: int, matrix_size: Vector2i) -> Error: 
		assert(matrix_size.y > 0)
		return copy_vector(src, dst, src_offset * matrix_size.y, dst_offset * matrix_size.y, 
			count * matrix_size.y, matrix_size.x)
	
	
	func get_data_async(callback: Callable, offset := 0, count := 0) -> Error: 
		return _ctx.rd.buffer_get_data_async(_buffer, callback, offset * FLOAT_SIZE, count * FLOAT_SIZE)
	
	
	func get_vector_data_async(callback: Callable, offset: int, count: int, vector_size: int) -> Error:
		return get_data_async(callback, offset * vector_size, count * vector_size)
	
	
	func get_matrix_data_async(callback: Callable, offset: int, count: int, matrix_size: Vector2i) -> Error:
		return get_vector_data_async(callback, offset * matrix_size.y, count * matrix_size.y, matrix_size.x)
	
	
	func get_data(offset := 0, count := 0) -> PackedByteArray:
		return _ctx.rd.buffer_get_data(_buffer, 0 * size, count * size) if _buffer else PackedByteArray()
	
	
	func _create_uniform_sets() -> void:
		if not _buffer:
			_read_uniform_set = RID()
			_write_uniform_set = RID()
		
		var uniform := RDUniform.new()
		uniform.binding = 0
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.add_id(_buffer)
		
		_read_uniform_set = _ctx.rd.uniform_set_create([uniform], _ctx._weight_shader, 0)
		_write_uniform_set = _ctx.rd.uniform_set_create([uniform], _ctx._weight_shader, 2)
