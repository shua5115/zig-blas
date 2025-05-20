//! matrix.zig
//! 
//! A straightforward matrix implementation designed to interface with CBLAS and LAPACK,
//! providing common packed matrix layouts.

const std = @import("std");

pub const z32 = std.math.Complex(f32);
pub const z64 = std.math.Complex(f64);
pub const Index = i32;
pub const DYNAMIC: Index = 0;

pub const MatF = Matrix(f32, DYNAMIC, DYNAMIC, .{});
pub const MatD = Matrix(f64, DYNAMIC, DYNAMIC, .{});
pub const MatCF = Matrix(z32, DYNAMIC, DYNAMIC, .{});
pub const MatCD = Matrix(z64, DYNAMIC, DYNAMIC, .{});
pub const Mat2F = Matrix(f32, 2, 2, .{});
pub const Mat2D = Matrix(f64, 2, 2, .{});
pub const Mat2CF = Matrix(z32, 2, 2, .{});
pub const Mat2CD = Matrix(z64, 2, 2, .{});
pub const Mat3F = Matrix(f32, 3, 3, .{});
pub const Mat3D = Matrix(f64, 3, 3, .{});
pub const Mat3CF = Matrix(z32, 3, 3, .{});
pub const Mat3CD = Matrix(z64, 3, 3, .{});
pub const Mat4F = Matrix(f32, 4, 4, .{});
pub const Mat4D = Matrix(f64, 4, 4, .{});
pub const Mat4CF = Matrix(z32, 4, 4, .{});
pub const Mat4CD = Matrix(z64, 4, 4, .{});

pub const VecF = Vector(f32, DYNAMIC);
pub const VecD = Vector(f64, DYNAMIC);
pub const VecCF = Vector(z32, DYNAMIC);
pub const VecCD = Vector(z64, DYNAMIC);
pub const Vec2F = Vector(f32, 2);
pub const Vec2D = Vector(f64, 2);
pub const Vec2CF = Vector(z32, 2);
pub const Vec2CD = Vector(z64, 2);
pub const Vec3F = Vector(f32, 3);
pub const Vec3D = Vector(f64, 3);
pub const Vec3CF = Vector(z32, 3);
pub const Vec3CD = Vector(z64, 3);
pub const Vec4F = Vector(f32, 4);
pub const Vec4D = Vector(f64, 4);
pub const Vec4CF = Vector(z32, 4);
pub const Vec4CD = Vector(z64, 4);

pub const Order = enum(u1) {
    ColMajor,
    RowMajor,
};

pub const Layout = enum(u7) {
    Dense,
    Symmetric,
    UpperTriangular,
    LowerTriangular,
    Diagonal,
    UpperBidiagonal,
    LowerBidiagonal,
    Tridiagonal,
};

pub const Storage = packed struct {
    order: Order = .ColMajor,
    layout: Layout = .Dense,
};

pub const MatrixError = error {
    DifferentSize,
    MissingAllocator,
    ZeroSize,
    NonSquare,
    NotVector,
};

pub fn is_complex_number(T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct") return false;
    const s = info.@"struct";
    if (s.fields.len != 2) return false;
    return std.mem.eql(u8, s.fields[0].name, "re") and std.mem.eql(u8, s.fields[1].name, "im");
}

pub fn matrix_unop(out: anytype, in: anytype, f: *const fn(out.Scalar) out.Scalar) void {
    for (0..@intCast(out.cols())) |c| {
        for (0..@intCast(out.rows())) |r| {
            const i: Index = @intCast(r);
            const j: Index = @intCast(c);
            out.set(i, j, f(@as(out.Scalar, in.get(i, j))));
        }
    }
}

pub fn matrix_unop_stateful(out: anytype, in: anytype, f: *const fn(*anyopaque, Index, Index, out.Scalar) out.Scalar, context: *anyopaque) void {
    for (0..@intCast(out.cols())) |c| {
        for (0..@intCast(out.rows())) |r| {
            const i: Index = @intCast(r);
            const j: Index = @intCast(c);
            out.set(i, j, f(context, i, j, @as(out.Scalar, in.get(i, j))));
        }
    }
}

pub fn matrix_binop(out: anytype, lhs: anytype, rhs: anytype, f: *const fn(out.Scalar, out.Scalar) out.Scalar) void {
    for (0..@intCast(out.cols())) |c| {
        for (0..@intCast(out.rows())) |r| {
            const i: Index = @intCast(r);
            const j: Index = @intCast(c);
            out.set(i, j, f(@as(out.Scalar, lhs.get(i, j)), @as(out.Scalar, rhs.get(i, j))));
        }
    }
}

pub fn matrix_binop_stateful(out: anytype, lhs: anytype, rhs: anytype, f: *const fn(*anyopaque, Index, Index, out.Scalar, out.Scalar) out.Scalar, context: *anyopaque) void {
    for (0..@intCast(out.cols())) |c| {
        for (0..@intCast(out.rows())) |r| {
            const i: Index = @intCast(r);
            const j: Index = @intCast(c);
            out.set(i, j, f(context, i, j, @as(out.Scalar, lhs.get(i, j)), @as(out.Scalar, rhs.get(i, j))));
        }
    }
}

/// Defines a Matrix type which contains a scalar type T.
/// T can be a float, int, or `std.math.Complex` of float or int.
/// 
/// Row and column sizes can be a fixed positive number, or set to DYNAMIC (0). If both row and column are fixed, the matrix is fixed-size.
/// 
/// NOTE: Fixed-size matrices are stored on the stack, so large matrices should be dynamic.
/// Dynamic matrix data is stored on the heap, referenced by a slice.
/// 
/// NOTE: Dynamic matrix operations without `anytype` arguments will panic if matrices aren't the same size.
/// Zig's error type system has been ignored for API uniformity with fixed-size matrices, which don't have this problem.
/// It is your responsibility to ensure dynamic matrices are the same size.
/// 
/// Storage by default is dense column major, and can be changed for more compact storage types.
/// Other storage types assert that some elements of a matrix are always zero or equal to other elements,
/// and take advantage of this by not storing redundant values.
/// - `storage.order` is ignored for Diagonal layouts, since data is stored in contiguous diagonals rather than rows and columns
/// - Symmetric layout is stored as upper triangular
pub fn Matrix(comptime T: type, comptime rows_fixed: Index, comptime cols_fixed: Index, comptime storage_type: Storage) type {
    if (comptime rows_fixed < 0 or cols_fixed < 0) @compileError("matrix size cannot be negative");
    const fixed_data_size = comptime storage_data_size(storage_type, rows_fixed, cols_fixed);
    const DataType = if (comptime rows_fixed > 0 and cols_fixed > 0)
        [fixed_data_size]T
     else
        []T;
    const SizeType = if (comptime rows_fixed > 0 and cols_fixed > 0)
            struct { rows: void, cols: void }
        else if (comptime rows_fixed > 0)
            struct { rows: void, cols: Index }
        else if (comptime cols_fixed > 0)
            struct { rows: Index, cols: void }
        else
            struct { rows: Index, cols: Index }
        ;
    if (comptime rows_fixed > 0 and cols_fixed > 0 and rows_fixed != cols_fixed) {
        switch(comptime storage_type.layout) {
            .Symmetric, .UpperTriangular, .LowerTriangular => @compileError("matrix must be square"),
            else => {}
        }
    }
    return struct {
        const Self = @This();

        /// Type stored in the matrix.
        pub const Scalar: type = T;
        /// Scalar representation of zero.
        pub const ZERO: Scalar = if (is_complex) Scalar{.re=0, .im=0} else 0;
        /// Scalar representation of one.
        pub const ONE: Scalar = if (is_complex) Scalar{.re=1, .im=0} else 1;
        /// If > 0, the comptime-known number of rows for this matrix type.
        pub const fixed_rows: Index = rows_fixed;
        /// If > 0, the comptime-known number of columns for this matrix type.
        pub const fixed_cols: Index = cols_fixed;
        /// If one of the matrix dimensions is dynamically sized.
        pub const is_dynamic: bool = rows_fixed <= 0 and cols_fixed <= 0;
        /// If this matrix stores a subtype of std.math.Complex.
        pub const is_complex: bool = is_complex_number(Scalar);
        /// Memory layout of this matrix type.
        pub const storage: Storage = storage_type;
        /// A range of indices in this matrix type.
        /// Defaults to cover the whole matrix.
        /// - NOTE: An end_row or end_col of 0 will be interpreted as the dynamic size of the matrix (rows(), cols() respectively)
        pub const Range = struct {
            start_row: Index = 0,
            start_col: Index = 0,
            end_row: Index = fixed_rows,
            end_col: Index = fixed_cols,
        };
        /// A view into this matrix type.
        /// Implements basic data access methods, but not arithmetic.
        /// Intended for use as input to other functions.
        pub const Slice = struct {
            ptr: *Self,
            range: Range,

            pub fn init(src: *Self, src_range: Range) Slice {
                const actual_range = Range{
                    .start_row = src_range.start_row,
                    .start_col = src_range.start_col,
                    .end_row = if (src_range.end_row == 0) src.rows() else src_range.end_row,
                    .end_col = if (src_range.end_col == 0) src.cols() else src_range.end_col,
                };
                return Slice{
                    .ptr = src,
                    .range = actual_range,
                };
            }

            pub fn rows(self: Slice) Index {
                return self.range.end_row-self.range.start_row;
            }

            pub fn cols(self: Slice) Index {
                return self.range.end_col-self.range.start_col;
            }

            fn data_index(self: Slice, i: Index, j: Index) ?struct{Index, Index} {
                if (i < 0 or j < 0 or i >= self.rows() or j >= self.cols()) return null;
                return .{ i + self.range.start_row, j + self.range.start_col };
            }

            pub fn get(self: Slice, i: Index, j: Index) Scalar {
                const rc = self.data_index(i, j) orelse return ZERO;
                return self.ptr.get(rc[0], rc[1]);
            }

            pub fn set(self: Slice, i: Index, j: Index, value: Scalar) void {
                const rc = self.data_index(i, j) orelse return;
                return self.ptr.set(rc[0], rc[1], value);
            }

            pub fn slice(self: Slice, range: struct {start_row: Index = 0, start_col: Index = 0, end_row: Index = 0, end_col: Index = 0}) Slice {
                const actual_range = Range{
                    .start_row = self.range.start_row + range.start_row,
                    .start_col = self.range.start_col + range.start_col,
                    .end_row = if (range.end_row == 0) self.range.end_row else self.range.start_row + range.end_row,
                    .end_col = if (range.end_col == 0) self.range.end_col else self.range.start_col + range.end_col,
                };
                return Slice{
                    .ptr = self.ptr,
                    .range = actual_range,
                };
            }

            pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
                _ = options.precision;
                const elem_fmt: []const u8 = if (comptime is_complex)
                    ("{" ++ fmt ++ "} + i*{" ++ fmt ++ "}\t")
                else
                    ("{" ++ fmt ++ "}\t")
                ;
                var i: Index = 0;
                while (i < self.rows()) {
                    var j: Index = 0;
                    while (j < self.cols()) {
                        const val = self.get(i, j);
                        if (comptime is_complex) {
                            try writer.print(elem_fmt, .{val.re, val.im});
                        } else {
                            try writer.print(elem_fmt, .{val});
                        }
                        j += 1;
                    }
                    try writer.writeByte('\n');
                    i += 1;
                }
            }
        };
        
        /// Stores the elements of this matrix.
        data: DataType = if(is_dynamic) undefined else [_]Scalar{ZERO} ** fixed_data_size,
        /// Stores the size of dynamic dimensions of the array.
        dyn_size: SizeType = undefined,

        /// checks if the matrix must be square based on its layout
        fn check_square(n_rows: Index, n_cols: Index) !void {
            switch(comptime storage_type.layout) {
                .Symmetric, .UpperTriangular, .LowerTriangular => if(n_rows != n_cols) return MatrixError.NonSquare,
                else => {}
            }
        }

        pub inline fn get_layout(self: Self) Layout {
            _ = self;
            return Self.storage.layout;
        }

        pub inline fn get_order(self: Self) Order {
            _ = self;
            return Self.storage.order;
        }

        /// Initializes a dynamically sized matrix and fills with ZERO.
        /// If the Matrix type is not dynamic, then the provided allocator can be null.
        pub fn init(allocator: ?std.mem.Allocator, n_rows: Index, n_cols: Index) !Self {
            if (comptime is_dynamic) {
                var m = try init_undefined(allocator, n_rows, n_cols);
                @memset(m.get_data(), ZERO);
                return m;
            }
            return .{};
        }

        pub fn init_vector(allocator: ?std.mem.Allocator, n_elems: Index) !Self {
            if (comptime is_dynamic) {
                var m: Self = try init_vector_undefined(allocator, n_elems);
                @memset(m.get_data(), ZERO);
                return m;
            }
            return .{};
        }

        /// Initializes a dynamically sized matrix and does not assign any data.
        /// If the matrix type is fixed size, then the provided allocator can be null.
        pub fn init_undefined(allocator: ?std.mem.Allocator, n_rows: Index, n_cols: Index) !Self {
            var m: Self = .{.data = undefined};
            if (comptime is_dynamic) {
                try check_square(n_rows, n_cols);
                if (comptime fixed_rows <= 0) {
                    m.dyn_size.rows = n_rows;
                }
                if (comptime fixed_cols <= 0) {
                    m.dyn_size.cols = n_cols;
                }
                if (m.rows() <= 0 or m.cols() <= 0) return MatrixError.ZeroSize;
                if (allocator == null) return MatrixError.MissingAllocator;
                m.data = try allocator.?.alloc(Scalar, storage_data_size(storage, @intCast(m.rows()), @intCast(m.cols())));
            }
            return m;
        }

        pub fn init_vector_undefined(allocator: ?std.mem.Allocator, n_elems: Index) !Self {
            var m: Self = .{.data = undefined};
            if (comptime is_dynamic) {
                if (comptime fixed_rows <= 0 and fixed_cols == 1) {
                    m.dyn_size.rows = n_elems;
                } else if (comptime fixed_cols <= 0 and fixed_rows == 1) {
                    m.dyn_size.cols = n_elems;
                } else {
                    return MatrixError.NotVector;
                }
                if (m.rows() <= 0 or m.cols() <= 0) return MatrixError.ZeroSize;
                if (allocator == null) return MatrixError.MissingAllocator;
                try check_square(m.rows(), m.cols());
                m.data = try allocator.?.alloc(Scalar, storage_data_size(storage, @intCast(m.rows()), @intCast(m.cols())));
            }
            return m;
        }

        pub fn init_identity(allocator: ?std.mem.Allocator, n_rows: Index, n_cols: Index) !Self {
            var m = try Self.init_undefined(allocator, n_rows, n_cols);
            m.assign_identity();
            return m;
        }

        /// Initializes a new matrix to be a copy of another matrix of the same type.
        /// If the Matrix type is not dynamic, then the allocator may be null.
        pub fn init_copy(allocator: ?std.mem.Allocator, src: Self) !Self {
            var m = try Self.init_undefined(allocator, src.rows(), src.cols());
            m.assign(src);
            return m;
        }

        /// Initializes a new matrix to be a copy of another matrix.
        /// `src` must have the following methods:
        /// - `fn get(self, i: Index, j: Index) Scalar`
        /// - `fn rows(self) Index`
        /// - `fn cols(self) Index`
        /// If the output Matrix type is not dynamic, then the allocator may be null.
        pub fn init_copy_any_matrix(allocator: ?std.mem.Allocator, src: anytype) !Self {
            var m = try Self.init_undefined(allocator, src.rows(), src.cols());
            m.assign_any_matrix(src);
            return m;
        }

        pub fn init_range_any_matrix(allocator: ?std.mem.Allocator, range: Range, src: anytype) !Self {
            var m = try Self.init_undefined(allocator, src.rows(), src.cols());
            m.assign_range_any_matrix(range, src);
            return m;
        }

        /// Initializes a new matrix to contain the contents of a 2D array or 2D slice.
        /// If arrays is smaller than matrix, missing values are zero.
        /// If arrays is larger than matrix, extra values are ignored.
        /// If the Matrix type is not dynamic, then the provided allocator can be null.
        pub fn init_arrays(allocator: ?std.mem.Allocator, arrays: anytype, arrays_order: Order) !Self {
            var m: Self = undefined;
            if (arrays_order == .RowMajor) {
                m = try init_undefined(allocator, arrays.len, arrays[0].len);
            } else {
                m = try init_undefined(allocator, arrays[0].len, arrays.len);
            }
            m.assign_arrays(arrays, arrays_order);
            return m;
        }

        /// Initializes a new matrix to contain the contents of a 2D nested tuple.
        /// If 2D tuple is smaller than matrix, missing values are zero.
        /// If 2D tuple is larger than matrix, extra values are ignored.
        /// The nested tuple is in row-major format.
        /// If the Matrix type is not dynamic, then the allocator may be null.
        pub fn init_tuples(allocator: ?std.mem.Allocator, tuples: anytype) !Self {
            var m = try init_undefined(allocator, tuples.len, tuples[0].len);
            m.assign_tuples(tuples);
            return m;
        }

        /// Initializes a new vector to contain the contents of an array or slice.
        /// If src is smaller than vector, missing values are zero.
        /// If src is larger than matrix, extra values are ignored.
        /// If the Vector type is not dynamic, then the provided allocator can be null.
        pub fn init_vector_array(allocator: ?std.mem.Allocator, src: anytype) !Self {
            var m: Self = try init_vector_undefined(allocator, src.len);
            m.vector_assign_array(src);
            return m;
        }

        /// Initializes a new vector to contain the contents of a tuple.
        /// If src is smaller than vector, missing values are zero.
        /// If src is larger than matrix, extra values are ignored.
        /// If the Vector type is not dynamic, then the provided allocator can be null.
        pub fn init_vector_tuple(allocator: ?std.mem.Allocator, src: anytype) !Self {
            var m: Self = try init_vector_undefined(allocator, src.len);
            m.vector_assign_tuple(src);
            return m;
        }

        /// If matrix is dynamic, this creates a dynamic matrix with its data stored in the given slice.
        /// - If the given slice is too small, this returns an error.
        /// - Calling `.resize(allocator, ...)` will only be legal if the provided slice was allocated by the same allocator
        /// 
        /// For fixed-sized matrices, copies data into the matrix, ignoring n_rows and n_cols.
        /// - If the slice is smaller than the fixed size, extra space is filled with zero.
        /// - If the slice is larger, extra values are ignored.
        pub fn init_wrap(data: []Scalar, n_rows: Index, n_cols: Index) !Self {
            var m: Self = undefined;
            const size = storage_data_size(storage, @intCast(n_rows), @intCast(n_cols));
            if (comptime is_dynamic) {
                try check_square(n_rows, n_cols);
                if (data.len < size) {
                    return MatrixError.DifferentSize;
                }
                m = Self{
                    .data = data,
                };
                if (comptime Self.fixed_rows <= 0) {
                    m.dyn_size.rows = n_rows;
                }
                if (comptime Self.fixed_cols <= 0) {
                    m.dyn_size.cols = n_cols;
                }
                return m;
            }
            m = Self{};
            const minsize = @min(size, data.len);
            @memcpy(m.get_data()[0..minsize], data[0..minsize]);
            return m;
        }

        pub fn copy(self: Self, allocator: ?std.mem.Allocator) !Self {
            return Self.init_copy(allocator, self);
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            if (comptime is_dynamic) {
                allocator.free(self.data);
            }
        }

        /// Resizes a matrix to a new size. Matrix data should be considered undefined after this operation.
        /// If n_rows or n_cols is null, then that dimension is not changed.
        /// If the dimension to change is fixed, then the change in that dimension is ignored.
        /// - For example, for a dynamic vector `Matrix(DYNAMIC, 1)`, calling `.resize(allocator, 2, 3)` ignores the 3, since the second dimension is fixed.
        pub fn resize(self: *Self, allocator: std.mem.Allocator, n_rows: ?Index, n_cols: ?Index) !void {
            if (comptime is_dynamic) {
                var new_rows = self.rows();
                var new_cols = self.cols();
                if (comptime fixed_rows <= 0) {
                    if (n_rows != null) {
                        if (n_rows.? <= 0) return MatrixError.ZeroSize;
                        new_rows = n_rows.?;
                    }
                }
                if (comptime fixed_cols <= 0) {
                    if (n_cols != null) {
                        if (n_cols.? <= 0) return MatrixError.ZeroSize;
                        new_cols = n_cols.?;
                    }
                }
                try check_square(new_rows, new_cols);
                if (comptime fixed_rows <= 0) {
                    self.dyn_size.rows = new_rows;
                }
                if (comptime fixed_cols <= 0) {
                    self.dyn_size.cols = new_cols;
                }
                const size = storage_data_size(storage, @intCast(self.rows()), @intCast(self.cols()));
                self.data = try allocator.realloc(self.data, size);
            }
        }

        /// Gets the number of rows in this matrix.
        pub fn rows(self: Self) Index {
            if (comptime fixed_rows <= 0) {
                return self.dyn_size.rows;
            } else return fixed_rows;
        }

        /// Gets the number of cols in this matrix.
        pub fn cols(self: Self) Index {
            if (comptime fixed_cols <= 0) {
                return self.dyn_size.cols;
            } else return fixed_cols;
        }

        pub fn is_vector(self: Self) bool {
            return if (comptime fixed_cols == 1) true
            else if (comptime fixed_rows == 1) true
            else if (self.cols() == 1) true
            else if (self.rows() == 1) true
            else false;
        }

        /// Gets the length of the vector. Panics if the matrix is not a vector.
        pub fn vector_size(self: Self) usize {
            const r: usize = @intCast(self.rows());
            const c: usize = @intCast(self.cols());
            if (comptime fixed_cols == 1) return r
            else if (comptime fixed_rows == 1) return c
            else if (self.cols() == 1) return r
            else if (self.rows() == 1) return c
            else @panic("not a vector");
        }

        /// Get the index for this row and column in the data storage.
        pub fn flat_index(self: Self, i: Index, j: Index) usize {
            return storage_index(storage, @intCast(self.rows()), @intCast(self.cols()), i, j);
        }

        /// Gets the (row, column) indices from an index into the matrix data.
        pub fn unflat_index(self: Self, index: usize) struct{Index, Index} {
            const r: usize = @intCast(self.rows());
            const c: usize = @intCast(self.cols());
            const size = storage_data_size(storage, r, c);
            return storage_rowcol(storage, r, c, size, index);
        }

        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = options.precision;
            const elem_fmt: []const u8 = if (comptime is_complex)
                ("{" ++ fmt ++ "} + i*{" ++ fmt ++ "}\t")
            else
                ("{" ++ fmt ++ "}\t")
            ;
            var i: Index = 0;
            while (i < self.rows()) {
                var j: Index = 0;
                while (j < self.cols()) {
                    const val = self.get(i, j);
                    if (comptime is_complex) {
                        try writer.print(elem_fmt, .{val.re, val.im});
                    } else {
                        try writer.print(elem_fmt, .{val});
                    }
                    j += 1;
                }
                try writer.writeByte('\n');
                i += 1;
            }
        }

        /// Gets the underlying matrix data slice.
        pub inline fn get_data(self: *Self) []Scalar {
            return self.data[0..];
        }

        pub inline fn get_const_data(self: *const Self) []const Scalar {
            return self.data[0..];
        }

        /// Gets a pointer to the matrix data.
        pub inline fn get_ptr(self: *Self) [*]Scalar {
            return self.get_data().ptr;
        }

        pub inline fn get_const_ptr(self: *const Self) [*]const Scalar {
            return self.get_const_data().ptr;
        }

        pub inline fn get_upper_diagonal(self: *Self) []Scalar {
            const n: usize = @intCast(@min(self.rows(), self.cols()));
            switch (comptime storage.layout) {
                .Tridiagonal, .UpperBidiagonal => return self.get_data()[0..n-1],
                else => @compileError("matrix does not have diagonal storage layout or does not store an upper diagonal")
            }
        }

        pub inline fn get_diagonal(self: *Self) []Scalar {
            const n: usize = @intCast(@min(self.rows(), self.cols()));
            switch (comptime storage.layout) {
                .Tridiagonal, .UpperBidiagonal => return self.get_data()[n-1..2*n-1],
                .LowerBidiagonal => return self.get_data()[0..n],
                else => @compileError("matrix does not have diagonal storage layout")
            }
        }

        pub inline fn get_lower_diagonal(self: *Self) []Scalar {
            const n: usize = @intCast(@min(self.rows(), self.cols()));
            switch (comptime storage.layout) {
                .Tridiagonal => return self.get_data()[2*n-1..3*n-2],
                .LowerBidiagonal => return self.get_data()[n..2*n-1],
                else => @compileError("matrix does not have diagonal storage layout or does not store a lower diagonal")
            }
        }

        /// Gets a single value at row i and column j.
        /// If the index is out of bounds, return ZERO.
        pub fn get(self: Self, i: Index, j: Index) Scalar {
            const idx = self.flat_index(i, j);
            if (idx >= self.data.len) {
                return ZERO;
            }
            return self.data[idx];
        }

        /// Sets a single value at row i and column j.
        /// If the index is out of bounds, nothing happens.
        pub fn set(self: *Self, i: Index, j: Index, value: Scalar) void {
            const idx = self.flat_index(i, j);
            if (idx < self.data.len) {
                self.data[idx] = value;
            }
        }

        pub fn vector_get(self: *Self, i: Index) Scalar {
            if (comptime fixed_cols == 1) return self.get(i, 0)
            else if (comptime fixed_rows == 1) return self.get(0, i)
            else if (self.cols() == 1) return self.get(i, 0)
            else if (self.rows() == 1) return self.get(0, i)
            else @panic("not a vector");
        }

        pub fn vector_set(self: *Self, i: Index, value: Scalar) void {
            if (comptime fixed_cols == 1) return self.set(i, 0, value)
            else if (comptime fixed_rows == 1) return self.set(0, i, value)
            else if (self.cols() == 1) return self.set(i, 0, value)
            else if (self.rows() == 1) return self.set(0, i, value)
            else @panic("not a vector");
        }

        pub fn slice(self: *Self, range: Range) Slice {
            return Slice.init(self, range);
        }

        pub fn fill(self: *Self, value: Scalar) void {
            @memset(self.get_data(), value);
        }

        /// Copies data from other into self.
        /// If not large enough to fill the whole matrix, unassigned elements are zero.
        /// If larger than the matrix, extra values are ignored.
        pub fn assign(self: *Self, other: Self) void {
            const min_size = @min(self.data.len, other.data.len);
            @memcpy(self.get_data()[0..min_size], other.get_const_data()[0..min_size]);
            if (min_size < self.data.len) {
                @memset(self.get_data()[min_size..], ZERO);
            }
        }

        pub fn assign_identity(self: *Self) void {
            const minsize = @min(self.rows(), self.cols());
            @memset(self.get_data(), ZERO);
            var i: Index = 0;
            while (i < minsize) {
                self.set(i, i, ONE);
                i += 1;
            }
        }

        /// Assigns to every element of this matrix from the src matrix.
        /// The other must have a `get` method with the following prototype: `pub fn get(self, i: Index, j: Index) Scalar`.
        pub fn assign_any_matrix(self: *Self, src: anytype) void {
            for (0..@intCast(self.cols())) |j| {
                for (0..@intCast(self.rows())) |i| {
                    self.set(@intCast(i), @intCast(j), @as(Scalar, src.get(@intCast(i), @intCast(j))));
                }
            }
        }

        /// Assigns to all elements in range from src.
        /// The src must have a `get` method with the following prototype: `pub fn get(self, i: Index, j: Index) Scalar`.
        pub fn assign_range_any_matrix(self: *Self, range: Range, src: anytype) void {
            const from_row: usize = @intCast(range.start_row);
            const to_row: usize = @intCast(if (range.end_row == 0) self.rows() else range.end_row);
            const from_col: usize = @intCast(range.start_col);
            const to_col: usize = @intCast(if (range.end_col == 0) self.cols() else range.end_col);
            for (from_col..to_col) |c| {
                for (from_row..to_row) |r| {
                    const i: Index = @intCast(r);
                    const j: Index = @intCast(c);
                    self.set(i, j, @as(Scalar, src.get(i, j)));
                }
            }
        }

        /// Assigns to every element of this matrix from a 2D array, or 2D nested slices.
        /// Order describes the 2D array layout.
        /// If not large enough to fill the whole matrix, unassigned elements are zero.
        /// If larger than the matrix, extra values are ignored.
        pub fn assign_arrays(self: *Self, arrays: anytype, arrays_order: Order) void {
            @memset(self.get_data(), ZERO);
            if (arrays_order == .RowMajor) {
                for (arrays, 0..) |row, i| {
                    if (i >= self.rows()) break;
                    for (row, 0..) |val, j| {
                        if (j >= self.cols()) break;
                        self.set(@intCast(i), @intCast(j), @as(Scalar, val));
                    }
                }
            } else {
                for (arrays, 0..) |col, j| {
                    if (j >= self.cols()) break;
                    for (col, 0..) |val, i| {
                        if (i >= self.rows()) break;
                        self.set(@intCast(i), @intCast(j), @as(Scalar, val));
                    }
                }
            }
        }

        /// Assigns data to this matrix, where tuples is 2D array of tuples in row-major format.
        /// If not large enough to fill the whole matrix, unassigned elements are zero.
        /// If larger than the matrix, extra values are ignored.
        pub fn assign_tuples(self: *Self, tuples: anytype) void {
            const TuplesType = @TypeOf(tuples);
            const info = @typeInfo(TuplesType);
            if (comptime info != .@"struct" or !info.@"struct".is_tuple) @compileError("not a tuple");
            @memset(self.get_data(), ZERO);
            inline for (tuples, 0..) |row, i| {
                if (comptime (!is_dynamic) and i >= fixed_rows) break;
                inline for (row, 0..) |val, j| {
                    if (comptime (!is_dynamic) and j >= fixed_cols) break;
                    self.set(@intCast(i), @intCast(j), @as(Scalar, val));
                }
            }
        }

        /// Assigns to every element of this vector from an array or slice.
        /// If not large enough to fill the whole matrix, unassigned elements are zero.
        /// If larger than the matrix, extra values are ignored.
        pub fn vector_assign_array(self: *Self, src: anytype) void {
            @memset(self.get_data(), ZERO);
            const r: usize = @intCast(self.rows());
            const c: usize = @intCast(self.cols());
            if (comptime fixed_cols == 1) {
                for (src, 0..) |val, i| {
                    if (i >= r) break;
                    self.set(@intCast(i), 0, @as(Scalar, val));
                }
            } else if (comptime fixed_rows == 1) {
                for (src, 0..) |val, i| {
                    if (i >= c) break;
                    self.set(0, @intCast(i), @as(Scalar, val));
                }
            } else if (self.cols() == 1) {
                for (src, 0..) |val, i| {
                    if (i >= r) break;
                    self.set(@intCast(i), 0, @as(Scalar, val));
                }
            } else if (self.rows() == 1) {
                for (src, 0..) |val, i| {
                    if (i >= c) break;
                    self.set(0, @intCast(i), @as(Scalar, val));
                }
            } else @panic("not a vector");
        }

        /// Assigns to every element of this vector from a tuple.
        /// If not large enough to fill the whole matrix, unassigned elements are zero.
        /// If larger than the matrix, extra values are ignored.
        pub fn vector_assign_tuple(self: *Self, src: anytype) void {
            @memset(self.get_data(), ZERO);
            const r: usize = @intCast(self.rows());
            const c: usize = @intCast(self.cols());
            if (comptime fixed_cols == 1) {
                inline for (src, 0..) |val, i| {
                    if (i >= r) break;
                    self.set(@intCast(i), 0, @as(Scalar, val));
                }
            } else if (comptime fixed_rows == 1) {
                inline for (src, 0..) |val, i| {
                    if (i >= c) break;
                    self.set(0, @intCast(i), @as(Scalar, val));
                }
            } else if (self.cols() == 1) {
                inline for (src, 0..) |val, i| {
                    if (i >= r) break;
                    self.set(@intCast(i), 0, @as(Scalar, val));
                }
            } else if (self.rows() == 1) {
                inline for (src, 0..) |val, i| {
                    if (i >= c) break;
                    self.set(0, @intCast(i), @as(Scalar, val));
                }
            } else @panic("not a vector");
        }

        fn scalar_neg(s: Scalar) Scalar {
            if (is_complex) {
                return s.neg();
            }
            return -s;
        }

        fn scalar_add(a: Scalar, b: Scalar) Scalar {
            if (is_complex) {
                return a.add(b);
            }
            return a + b;
        }

        fn scalar_sub(a: Scalar, b: Scalar) Scalar {
            if (is_complex) {
                return a.sub(b);
            }
            return a - b;
        }

        fn scalar_mul(a: Scalar, b: Scalar) Scalar {
            if (is_complex) {
                return a.mul(b);
            }
            return a * b;
        }

        fn scalar_div(a: Scalar, b: Scalar) Scalar {
            if (is_complex) {
                return a.div(b);
            }
            return a / b;
        }

        fn scalar_pow(a: Scalar, b: Scalar) Scalar {
            if (is_complex) {
                return std.math.complex.pow(a, b);
            }
            return std.math.pow(a, b);
        }

        fn scalar_eql(a: Scalar, b: Scalar) bool {
            if (is_complex) {
                return a.re == b.re and a.im == b.im;
            }
            return a == b;
        }

        // Panics if dynamic matrices are not the same size.
        inline fn guard_same_size(a: *const Self, b: *const Self) void {
            if (comptime is_dynamic) {
                if (a.dyn_rows != b.dyn_rows or a.dyn_cols != b.dyn_cols) {
                    @panic("different matrix sizes");
                }
            }
        }

        /// Sets each element at (i,j) to `f(context, i, j self.get(i, j))`.
        pub fn cwiseop_stateful(self: *Self, f: *const fn(?*anyopaque, Index, Index, Scalar) Scalar, context: ?*anyopaque) void {
            const size = storage_data_size(storage, @intCast(self.rows()), @intCast(self.cols()));
            for (self.data, 0..) |s, i| {
                const rc = storage_rowcol(storage, @intCast(self.rows()), @intCast(self.cols()), size, i);
                self.data[i] = f(context, rc[0], rc[1], s);
            }
        }

        /// Sets each element at (i,j) to `f(self.get(i, j))`.
        pub fn cwiseop(self: *Self, f: *const fn(Scalar) Scalar) void {
            for (self.data, 0..) |s, i| {
                self.data[i] = f(s);
            }
        }

        /// Sets each element `out_ij` to `f(context, i, j, in_ij)`.
        pub fn unop_stateful(out: *Self, in: Self, f: *const fn(?*anyopaque, Index, Index, Scalar) Scalar, context: ?*anyopaque) void {
            guard_same_size(out, &in);
            const size = storage_data_size(storage, @intCast(out.rows()), @intCast(out.cols()));
            for (0..out.data.len) |i| {
                const rc = storage_rowcol(storage, @intCast(out.rows()), @intCast(out.cols()), size, i);
                out.data[i] = f(context, rc[0], rc[1], in.data[i]);
            }
        }

        /// Sets each element `out_ij` to `f(in_ij)`.
        pub fn unop(out: *Self, in: Self, f: *const fn(Scalar) Scalar) void {
            guard_same_size(out, &in);
            for (0..out.data.len) |i| {
                out.data[i] = f(in.data[i]);
            }
        }

        /// Sets each element `out_ij` to `f(context, lhs_ij, rhs_ij)`.
        pub fn binop_stateful(out: *Self, lhs: Self, rhs: Self, f: *const fn(?*anyopaque, Index, Index, Scalar, Scalar) Scalar, context: ?*anyopaque) void {
            guard_same_size(out, &lhs);
            guard_same_size(out, &rhs);
            const size = storage_data_size(storage, @intCast(out.rows()), @intCast(out.cols()));
            for (0..out.data.len) |i| {
                const rc = storage_rowcol(storage, @intCast(out.rows()), @intCast(out.cols()), size, i);
                out.data[i] = f(context, rc[0], rc[1], lhs.data[i], rhs.data[i]);
            }
        }

        /// Sets each element `out_ij` to `f(lhs_ij, rhs_ij)`.
        pub fn binop(out: *Self, lhs: Self, rhs: Self, f: *const fn(Scalar, Scalar) Scalar) void {
            guard_same_size(out, &lhs);
            guard_same_size(out, &rhs);
            for (0..out.data.len) |i| {
                out.data[i] = f(lhs.data[i], rhs.data[i]);
            }
        }

        /// Sets each element `out_ij` to `f(context, i, j, in_ij)`.
        pub fn unop_any_stateful(out: *Self, in: anytype, f: *const fn(?*anyopaque, Index, Index, Scalar) Scalar, context: ?*anyopaque) void {
            const size = storage_data_size(storage, @intCast(out.rows()), @intCast(out.cols()));
            for (0..out.data.len) |i| {
                const rc = storage_rowcol(storage, @intCast(out.rows()), @intCast(out.cols()), size, i);
                out.data[i] = f(context, rc[0], rc[1], in.get(rc[0], rc[1]));
            }
        }

        /// Sets each element `out_ij` to `f(in_ij)`.
        pub fn unop_any(out: *Self, in: anytype, f: *const fn(Scalar) Scalar) void {
            const size = storage_data_size(storage, @intCast(out.rows()), @intCast(out.cols()));
            for (0..out.data.len) |i| {
                const rc = storage_rowcol(storage, @intCast(out.rows()), @intCast(out.cols()), size, i);
                out.data[i] = f(in.get(rc[0], rc[1]));
            }
        }

        /// Sets each element `out_ij` to `f(context, lhs_ij, rhs_ij)`.
        pub fn binop_any_stateful(out: *Self, lhs: anytype, rhs: anytype, f: *const fn(?*anyopaque, Index, Index, Scalar, Scalar) Scalar, context: ?*anyopaque) void {
            const size = storage_data_size(storage, @intCast(out.rows()), @intCast(out.cols()));
            for (0..out.data.len) |i| {
                const rc = storage_rowcol(storage, @intCast(out.rows()), @intCast(out.cols()), size, i);
                out.data[i] = f(context, rc[0], rc[1], lhs.get(rc[0], rc[1]), rhs.get(rc[0], rc[1]));
            }
        }

        /// Sets each element `out_ij` to `f(lhs_ij, rhs_ij)`.
        pub fn binop_any(out: *Self, lhs: anytype, rhs: anytype, f: *const fn(Scalar, Scalar) Scalar) void {
            const size = storage_data_size(storage, @intCast(out.rows()), @intCast(out.cols()));
            for (0..out.data.len) |i| {
                const rc = storage_rowcol(storage, @intCast(out.rows()), @intCast(out.cols()), size, i);
                out.data[i] = f(lhs.get(rc[0], rc[1]), rhs.get(rc[0], rc[1]));
            }
        }

        pub fn add_scalar(self: *Self, value: Scalar) void {
            for (self.data, 0..) |s, i| {
                self.data[i] = scalar_add(s, value);
            }
        }

        pub fn sub_scalar(self: *Self, value: Scalar) void {
            for (self.data, 0..) |s, i| {
                self.data[i] = scalar_sub(s, value);
            }
        }

        pub fn mul_scalar(self: *Self, value: Scalar) void {
            for (self.data, 0..) |s, i| {
                self.data[i] = scalar_mul(s, value);
            }
        }

        pub fn div_scalar(self: *Self, value: Scalar) void {
            for (self.data, 0..) |s, i| {
                self.data[i] = scalar_div(s, value);
            }
        }

        pub fn neg(out: *Self, in: Self) void {
            unop(out, in, scalar_neg);
        }

        pub fn add(out: *Self, lhs: Self, rhs: Self) void {
            binop(out, lhs, rhs, scalar_add);
        }

        pub fn sub(out: *Self, lhs: Self, rhs: Self) void {
            binop(out, lhs, rhs, scalar_sub);
        }

        pub fn mul(out: *Self, lhs: Self, rhs: Self) void {
            binop(out, lhs, rhs, scalar_mul);
        }

        pub fn div(out: *Self, lhs: Self, rhs: Self) void {
            binop(out, lhs, rhs, scalar_div);
        }

        pub fn add_any(out: *Self, lhs: anytype, rhs: anytype) void {
            binop_any(out, lhs, rhs, scalar_add);
        }

        pub fn sub_any(out: *Self, lhs: anytype, rhs: anytype) void {
            binop_any(out, lhs, rhs, scalar_sub);
        }

        pub fn mul_any(out: *Self, lhs: anytype, rhs: anytype) void {
            binop_any(out, lhs, rhs, scalar_mul);
        }

        pub fn div_any(out: *Self, lhs: anytype, rhs: anytype) void {
            binop_any(out, lhs, rhs, scalar_div);
        }

        pub fn fold(self: Self, initial_value: Scalar, f: *const fn(Scalar, Scalar) Scalar) Scalar {
            var total: Scalar = initial_value;
            for (self.data) |s| {
                total = f(total, s);
            }
            return total;
        }

        pub fn sum(self: Self) Scalar {
            return fold(self, ZERO, scalar_add);
        }

        pub fn prod(self: Self) Scalar {
            return fold(self, ONE, scalar_mul);
        }

        pub fn dot(self: Self, other: Self) Scalar {
            guard_same_size(self, other);
            var total: Scalar = ZERO;
            for (0..self.data.len) |i| {
                total = scalar_add(total, scalar_mul(self.data[i], other.data[i]));
            }
            return total;
        }

        pub fn dot_any(self: Self, other: anytype) Scalar {
            const size = storage_data_size(storage, @intCast(self.rows()), @intCast(self.cols()));
            var total: Scalar = ZERO;
            for (0..self.data.len) |i| {
                const rc = storage_rowcol(storage, @intCast(self.rows()), @intCast(self.cols()), size, i);
                total = scalar_add(total, scalar_mul(self.data[i], other.get(rc[0], rc[1])));
            }
            return total;
        }

        pub fn eql(self: Self, other: Self) bool {
            if (self.rows() != other.rows() or self.cols() != other.cols()) return false;
            for (0..self.data.len) |i| {
                const mine = self.data[i];
                const theirs = other.data[i];
                if (!scalar_eql(mine, theirs)) {
                    return false;
                }
            }
            return true;
        }

        pub fn eql_approx(self: Self, other: Self, tol: f64) bool {
            if (self.rows() != other.rows() or self.cols() != other.cols()) return false;
            for (0..self.data.len) |i| {
                const mine = self.data[i];
                const theirs = other.data[i];
                const diff = scalar_sub(mine, theirs);
                if (comptime is_complex) {
                    if (std.math.complex.abs(diff) > @as(@TypeOf(diff.re), tol)) return false;
                } else {
                    if (@abs(diff) > @as(Scalar, tol)) return false;
                }
            }
            return true;
        }

        pub fn eql_any(self: Self, other: anytype) bool {
            if (self.rows() != other.rows() or self.cols() != other.cols()) return false;
            for (0..@intCast(self.cols())) |r| {
                for (0..@intCast(self.rows())) |c| {
                    const i: Index = @intCast(r);
                    const j: Index = @intCast(c);
                    const mine = self.get(i, j);
                    const theirs = other.get(i, j);
                    if (!scalar_eql(mine, theirs)) {
                        return false;
                    }
                }
            }
            return true;
        }

        pub fn eql_any_approx(self: Self, other: anytype, tol: f64) bool {
            if (self.rows() != other.rows() or self.cols() != other.cols()) return false;
            for (0..@intCast(self.cols())) |r| {
                for (0..@intCast(self.rows())) |c| {
                    const i: Index = @intCast(r);
                    const j: Index = @intCast(c);
                    const mine = self.get(i, j);
                    const theirs = other.get(i, j);
                    const diff = scalar_sub(mine, theirs);
                    if (comptime is_complex) {
                        if (std.math.complex.abs(diff) > @as(@TypeOf(diff.re), tol)) return false;
                    } else {
                        if (@abs(diff) > @as(Scalar, tol)) return false;
                    }
                }
            }
            return true;
        }
    };
}

pub fn Vector(comptime scalar: type, comptime size_fixed: Index) type {
    return Matrix(scalar, size_fixed, 1, .{});
}

pub fn RowVector(comptime scalar: type, comptime size_fixed: Index) type {
    return Matrix(scalar, 1, size_fixed, .{.order = .RowMajor});
}

/// Calculates the data size for a storage type given desired rows and columns.
inline fn storage_data_size(comptime storage: Storage, rows: usize, cols: usize) usize {
    if (rows <= 0 or cols <= 0) {
        return 0;
    }
    switch (comptime storage.layout) {
        Layout.Dense => return rows*cols,
        Layout.UpperTriangular, Layout.LowerTriangular, Layout.Symmetric => {
            const n: usize = @min(rows, cols);
            return n*(n+1)/2;
        },
        Layout.Tridiagonal => {
            const n: usize = @min(rows, cols);
            return 3*n-2;
        },
        Layout.UpperBidiagonal, Layout.LowerBidiagonal => {
            const n: usize = @min(rows, cols);
            return 2*n-1;
        },
        Layout.Diagonal => {
            return @min(rows, cols);
        },
    }
    unreachable;
}

/// Calculates the index in a storage type's data slice given the Matrix size.
/// May return an invalid index if i or j are out of bounds.
inline fn storage_index(comptime storage: Storage, rows: usize, cols: usize, r: Index, c: Index) usize {
    if (r < 0 or c < 0) {
        // invalid in any storage layout
        return rows*cols;
    }
    const i: usize = @intCast(r);
    const j: usize = @intCast(c);
    if (i >= rows or j >= cols) {
        // invalid in any storage layout
        return rows*cols;
    }
    switch(comptime storage.layout) {
        Layout.Dense => switch(comptime storage.order) {
            Order.ColMajor => return i + rows*j,
            Order.RowMajor => return i*cols + j,
        },
        Layout.Symmetric => if (i <= j) switch (comptime storage.order) {
            Order.ColMajor => return i+j*(j+1)/2,
            Order.RowMajor => return i*(2*cols-i-1)/2 + j,
        } else switch (comptime storage.order) {
            Order.ColMajor => return j+i*(i+1)/2,
            Order.RowMajor => return j*(2*cols-j-1)/2 + i,
        },
        Layout.UpperTriangular => if (i > j) { return rows*cols; } else { switch (comptime storage.order) {
            Order.ColMajor => return i+j*(j+1)/2,
            Order.RowMajor => return i*(2*cols-i-1)/2 + j,
        }},
        Layout.LowerTriangular => if (i < j) { return rows*cols; } else { switch (comptime storage.order) {
            Order.ColMajor => return i + j*(2*rows-j-1)/2,
            Order.RowMajor => return i*(i+1)/2 + j,
        }},
        Layout.Tridiagonal => {
            // [n-1] ++ [n] ++ [n-1]
            const n = @min(rows, cols);
            if (i == j) {
                return i+(n-1);
            } else if (i+1 == j) {
                return i;
            } else if (j+1 == i) {
                return j+(2*n-1);
            } else {
                // invalid index
                return 3*n-2;
            }
        },
        Layout.UpperBidiagonal => if (i > j) { return rows*cols; } else {
            // [n-1] ++ [n]
            const n = @min(rows, cols);
            if (i == j) {
                return i+(n-1);
            } else if (i+1 == j) {
                return i;
            } else {
                return 2*n-1;
            }
        },
        Layout.LowerBidiagonal => if (i < j) { return rows*cols; } else {
            // [n] ++ [n-1]
            const n = @min(rows, cols);
            if (i == j) {
                return j;
            } else if (j+1 == i) {
                return j+n;
            } else {
                return 2*n-1;
            }
        },
        Layout.Diagonal => {
            if (i == j) {
                return i;
            }
            return @min(rows, cols);
        },
    }
    unreachable;
}

/// Calculates the row and column in a Matrix given an index into the storage type's data slice.
/// Returns (-1, -1) if index is out of bounds.
inline fn storage_rowcol(comptime storage: Storage, rows: usize, cols: usize, size: usize, index: usize) struct{Index, Index} {
    // const size = storage_data_size(storage, rows, cols);
    if (index >= size) {
        return .{-1, -1};
    }
    switch(comptime storage.layout) {
        Layout.Dense => switch(comptime storage.order) {
            Order.ColMajor => return .{ @intCast(index%rows), @intCast(index/rows) },
            Order.RowMajor => return .{ @intCast(index/cols), @intCast(index%cols) },
        },
        // be thankful this is done for you, this triangular number bs is confusing
        // Upper colmajor: https://www.desmos.com/calculator/b7eglfzqh7
        // Upper rowmajor: https://www.desmos.com/calculator/yyk7wskshl
        // Lower colmajor: https://www.desmos.com/calculator/fwydhcisal
        // Lower rowmajor: https://www.desmos.com/calculator/nsrceb0odu
        Layout.UpperTriangular, Layout.Symmetric => switch (comptime storage.order) {
            Order.ColMajor => {
                const x: usize = index;
                const k: usize = (std.math.sqrt(1 + 8*x) - 1)/2;
                return .{ @intCast(x - k*(k+1)/2), @intCast(k) };
            },
            Order.RowMajor => {
                const x: usize = index;
                const c: usize = cols*(cols+1)/2;
                const k: usize = (std.math.sqrt(1 + 8*(c-x-1)) - 1)/2;
                return .{ @intCast(cols - 1 - k), @intCast(x + cols + k*(k+1)/2 - c) };
            },
        },
        Layout.LowerTriangular => switch (comptime storage.order) {
            Order.ColMajor => {
                const x: usize = index;
                const r: usize = rows*(rows+1)/2;
                const k: usize = (std.math.sqrt(1 + 8*(r-x-1)) - 1)/2;
                return .{ @intCast(x + rows + k*(k+1)/2 - r), @intCast(rows - 1 - k) };
            },
            Order.RowMajor => {
                const x: usize = index;
                const k: usize = (std.math.sqrt(1 + 8*x) - 1)/2;
                return .{ @intCast(k), @intCast(x - k*(k+1)/2) };
            },
        },
        Layout.Tridiagonal => {
            const n: usize = @min(rows, cols);
            if (index < n-1) {
                return .{ @intCast(index), @intCast(index+1) };
            } else if (index < 2*n-1) {
                const i = index - (n-1);
                return .{ @intCast(i), @intCast(i) };
            } else if (index < 3*n-2) {
                const i = index - (2*n-1);
                return .{ @intCast(i+1), @intCast(i) };
            } else {
                return .{ -1, -1 };
            }
        },
        Layout.UpperBidiagonal => {
            const n = @min(rows, cols);
            if (index < n-1) {
                return .{ @intCast(index), @intCast(index+1) };
            } else if (index < 2*n-1) {
                const i = index - (n-1);
                return .{ @intCast(i), @intCast(i) };
            } else {
                return .{ -1, -1 };
            }
        },
        Layout.LowerBidiagonal => {
            const n = @min(rows, cols);
            if (index < n) {
                return .{ @intCast(index), @intCast(index) };
            } else if (index < 2*n-1) {
                const i = index - n;
                return .{ @intCast(i+1), @intCast(i) };
            } else {
                return .{ -1, -1 };
            }
        },
        Layout.Diagonal => {
            return .{@intCast(index), @intCast(index)};
        },
    }
    unreachable;
}

test "assign" {
    const alloc = std.testing.allocator;

    var A = Mat2D{};
    try std.testing.expectEqual(A.data, .{0, 0, 0, 0});
    A.assign_tuples(.{
        .{1, 2},
        .{3, 4},
    });
    try std.testing.expect(A.get(0, 0) == 1.0);
    try std.testing.expect(A.get(0, 1) == 2.0);
    try std.testing.expect(A.get(1, 0) == 3.0);
    try std.testing.expect(A.get(1, 1) == 4.0);

    const data = [2][2]f64{
        [_]f64{5, 6},
        [_]f64{7, 8},
    };
    var B = try MatD.init(alloc, 2, 2);
    defer B.deinit(alloc);
    B.assign_arrays(data, .RowMajor);
    try std.testing.expect(B.get(0, 0) == 5.0);
    try std.testing.expect(B.get(0, 1) == 6.0);
    try std.testing.expect(B.get(1, 0) == 7.0);
    try std.testing.expect(B.get(1, 1) == 8.0);
    A.assign_any_matrix(B);
    try std.testing.expect(A.get(0, 0) == 5.0);
    try std.testing.expect(A.get(0, 1) == 6.0);
    try std.testing.expect(A.get(1, 0) == 7.0);
    try std.testing.expect(A.get(1, 1) == 8.0);
}

fn check_unflat(m: anytype) !void {
    for (0..@intCast(m.cols())) |c| {
        for (0..@intCast(m.rows())) |r| {
            const i: Index = @intCast(r);
            const j: Index = @intCast(c);
            const index = m.flat_index(i, j);
            if (index >= m.data.len) continue;
            const rc = m.unflat_index(index);
            try std.testing.expectEqualDeep(@TypeOf(rc){i, j}, rc);
        }
    }
}

// ------- TESTS --------

test "layouts" {
    const UT = Matrix(f32, 3, 3, .{.layout = .UpperTriangular});
    var A = UT{};
    try std.testing.expectEqual(6, storage_data_size(UT.storage, 3, 3));
    A.assign_tuples(.{
        .{1, 2, 4},
        .{0, 3, 5},
        .{0, 0, 6},
    });
    A.set(1, 0, 69); // this shoud do nothing, since this index is not stored
    try std.testing.expectEqual(A.get(1, 0), 0.0);
    inline for (A.data, 0..) |s, i| {
        try std.testing.expectEqual(i+1, s);
    }
    try check_unflat(A);

    const LT = Matrix(f32, 3, 3, .{.layout = .LowerTriangular});
    var B = LT{};
    B.assign_tuples(.{
        .{1, 0, 0},
        .{2, 4, 0},
        .{3, 5, 6},
    });
    B.set(0, 1, 69); // this shoud do nothing, since this index is not stored
    try std.testing.expectEqual(0.0, B.get(0, 1));
    inline for (B.data, 0..) |s, i| {
        try std.testing.expectEqual(i+1, s);
    }
    try check_unflat(B);

    const TD = Matrix(f32, 3, 3, .{.layout = .Tridiagonal});
    var C = TD{};
    C.assign_tuples(.{
        .{3, 1, 0},
        .{6, 4, 2},
        .{0, 7, 5},
    });
    C.set(0, 2, 69); // this shoud do nothing, since this index is not stored
    try std.testing.expectEqual(0.0, C.get(0, 2));
    inline for (C.data, 0..) |s, i| {
        try std.testing.expectEqual(i+1, s);
    }
    try check_unflat(C);

    const UBD = Matrix(f32, 3, 3, .{.layout = .UpperBidiagonal});
    var D = UBD{};
    D.assign_tuples(.{
        .{3, 1, 0},
        .{0, 4, 2},
        .{0, 0, 5},
    });
    D.set(2, 0, 69); // this shoud do nothing, since this index is not stored
    try std.testing.expectEqual(0.0, D.get(2, 0));
    inline for (D.data, 0..) |s, i| {
        try std.testing.expectEqual(i+1, s);
    }
    try check_unflat(D);

    const LBD = Matrix(f32, 3, 3, .{.layout = .LowerBidiagonal});
    var E = LBD{};
    E.assign_tuples(.{
        .{1, 0, 0},
        .{4, 2, 0},
        .{0, 5, 3},
    });
    E.set(0, 2, 69); // this shoud do nothing, since this index is not stored
    try std.testing.expectEqual(0.0, E.get(0, 2));
    inline for (E.data, 0..) |s, i| {
        try std.testing.expectEqual(i+1, s);
    }
    try check_unflat(E);

    const SYM = Matrix(f32, 3, 3, .{.layout = .Symmetric});
    var F = SYM{};
    F.set(0, 0, 1);
    F.set(0, 1, 2);
    F.set(1, 1, 3);
    F.set(0, 2, 4);
    F.set(1, 2, 5);
    F.set(2, 2, 6);
    var F_expected = Mat3F{};
    F_expected.assign_tuples(.{
        .{1, 2, 4},
        .{2, 3, 5},
        .{4, 5, 6},
    });
    for (0..@intCast(F.rows())) |r| {
        for (0..@intCast(F.cols())) |c| {
            const i: Index = @intCast(r);
            const j: Index = @intCast(c);
            // std.debug.print("({},{}) => [{}] = {}\n", .{i, j, F.flat_index(i, j), F.get(i, j)});
            try std.testing.expectEqual(F_expected.get(i, j), F.get(i, j));
        }
    }
    // try check_unflat(E); // fails because flat_index and unflat_index are not 1 to 1 maps for symmetric matrices
    
    const DG = Matrix(f32, 3, 3, .{ .layout = .Diagonal });
    var G = DG{};
    G.assign_tuples(.{
        .{1, 0, 0},
        .{0, 2, 0},
        .{0, 0, 3}
    });
    G.set(0, 2, 69); // this shoud do nothing, since this index is not stored
    try std.testing.expectEqual(0.0, G.get(0, 2));
    inline for (G.data, 0..) |s, i| {
        try std.testing.expectEqual(i+1, s);
    }
    try check_unflat(G);
}

test "arith" {
    var A = Mat3D{};
    A.assign_tuples(.{
        .{1, 2, 3},
        .{4, 5, 6},
        .{7, 8, 9}
    });
    var B = Mat3D{};
    B.assign_tuples(.{
        .{9, 8, 7},
        .{6, 5, 4},
        .{3, 2, 1}
    });
    var E = Mat3D{};
    E.fill(10);
    var C = Mat3D{};
    C.add(A, B);
    try std.testing.expectEqual(E.data, C.data);
    C.add_any(A, B);
    try std.testing.expectEqual(E.data, C.data);
    C.add_any(&A, &B);
    try std.testing.expectEqual(E.data, C.data);

    E.assign_tuples(.{
        .{9, 16, 21},
        .{24, 25, 24},
        .{21, 16, 9}
    });
    C.mul(A, B);
    try std.testing.expectEqual(E.data, C.data);
    C.mul_any(A, B);
    try std.testing.expectEqual(E.data, C.data);

    var V = Vec3D{};
    V.vector_assign_tuple(.{1, 4, 7});
    C.sub_any(A, V);
    try std.testing.expectEqual(0, C.get(0,0));
    try std.testing.expectEqual(0, C.get(1,0));
    try std.testing.expectEqual(0, C.get(2,0));
}

test "slice" {
    var A = Mat3D{};
    A.assign_tuples(.{
        .{1, 2, 3},
        .{4, 5, 6},
        .{7, 8, 9}
    });

    var B = Mat2D{};
    B.assign_any_matrix(A.slice(.{.start_row = 1, .start_col = 1, .end_row = 3, .end_col = 3}));
    var E = Mat2D{};
    E.assign_tuples(.{
        .{5, 6},
        .{8, 9}
    });
    try std.testing.expectEqual(E.data, B.data);
}

test "pointers" {
    const A = try Mat3D.init_tuples(null, .{
        .{1, 2, 3},
        .{4, 5, 6},
        .{7, 8, 9}
    });

    const aptr: [*c]const f64 = A.get_const_ptr();
    // why is this valid syntax
    try std.testing.expectEqual(1.0, aptr.?.?.?.?.?.?.?.*);

    var B = try MatD.init_tuples(std.testing.allocator, .{
        .{1, 2, 3},
        .{4, 5, 6},
        .{7, 8, 9},
    });
    defer B.deinit(std.testing.allocator);
    B.add_any(A, B);
    const bptr: [*c]const f64 = B.get_const_ptr();
    try std.testing.expectEqual(2.0, bptr.*);
}