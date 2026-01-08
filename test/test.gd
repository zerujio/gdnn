extends Control


func _ready() -> void:
	test_interpolate_indexed()


func test_interpolate_indexed() -> void:
	const GROUP_SIZE_LOG2 := 4
	const GROUP_SIZE := 2 ** GROUP_SIZE_LOG2
	const GROUP_COUNT := 8
	const ELEMENT_COUNT := GROUP_COUNT * GROUP_SIZE
	
	var data := random_array_f32(ELEMENT_COUNT)
	var a := random_array_f32(ELEMENT_COUNT)
	var idx := random_array_i32(GROUP_COUNT * 2)
	
	var data_buf := GlobalNNContext.create_array_buffer(data.size(), data.to_byte_array())
	var idx_buf := GlobalNNContext.create_array_buffer(idx.size(), idx.to_byte_array())
	var a_buf := GlobalNNContext.create_array_buffer(a.size(), a.to_byte_array())
	var result_buf := GlobalNNContext.create_array_buffer(ELEMENT_COUNT)
	
	GlobalNNContext.interpolate_indexed(data_buf, data_buf, a_buf, idx_buf, GROUP_SIZE_LOG2, result_buf)
	
	result_buf.get_data_async(func(d: PackedByteArray):
		var expected := interpolate_indexed(data, data, a, idx, GROUP_SIZE)
		print("expected: ", expected)
		var actual := d.to_float32_array()
		print("actual: ", actual)
		
		for i in range(expected.size()):
			var e_i := expected[i]
			var a_i := actual[i]
			assert(is_equal_approx(e_i, a_i), 
				"expected[%d] (%f) != actual[%d] (%f)" % [i, e_i, i, a_i])
	)


func interpolate_indexed(x: PackedFloat32Array, y: PackedFloat32Array, a: PackedFloat32Array, 
		indices: PackedInt32Array, index_divisor: int) -> PackedFloat32Array:
		var result: PackedFloat32Array
		var index_count := indices.size() >> 1
		result.resize(index_count * index_divisor)
		for i in range(index_count):
			var x_idx := indices[2 * i]
			var y_idx := indices[2 * i + 1]
			for j in range(index_divisor):
				var k := i * index_divisor + j
				result[k] = lerpf(y[y_idx * index_divisor + j], x[x_idx * index_divisor + j], a[k])
		
		return result


func random_array_f32(count: int, from := 0.0, to := 1.0) -> PackedFloat32Array:
	var f: PackedFloat32Array
	f.resize(count)
	for i in range(f.size()):
		f[i] = randf_range(from, to)
	return f


func random_array_i32(count: int, from := 0, to := 1) -> PackedInt32Array:
	var i: PackedInt32Array
	i.resize(count)
	for j in range(i.size()):
		i[j] = randi_range(from, to)
	return i
