# Basic Linear Algebra Subprograms (BLAS) Wrapper for Zig

A thin wrapper around CBLAS, providing some API conveniences:
- Default arguments for common use cases, allowing most BLAS calls to be shorter
- A matrix implementation for common packed storage layouts
- Doc comments on every BLAS function for a quick reference on what they do

For more information about BLAS, see the [BLAS Quick Reference Guide](blas.pdf) and [netlib.org](https://www.netlib.org/blas/faq.html).

A generic Matrix type is implemented in [`matrix.zig`](src/matrix.zig).
The implementation is designed to take advantage of various packed matrix layouts used in CBLAS and LAPACK.
The file has no dependencies, so it does not need to be linked to CBLAS or LAPACK.

The generic Matrix type can store any numeric or complex numeric type, can have a fixed or dynamic size, and only stores the runtime variables necessary to track its size.
This is similar to the way the [Eigen Matrix Class](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html) works in C++.

Because the goal is to interface with CBLAS, only basic componentwise operations are implemented for the Matrix type.
CBLAS can be used for more complex operations.

### BLAS Function Defaults
- Order: Column-Major
- Transpose ("trans"): None, matrices will not be transposed by default
- Diagonal ("diag"): Non-Unit, diagonal entries may not be 1 by default
- Leading dimension ("ld*") defaults to 0. Any "ld*" value <= 0 is instead computed assuming the matrix is a 2D array of appropriate size and layout.
    - This only needs to be changed when working with subsections of a matrix, or a matrix where data is padded after every row/column.

### Supported Matrix Layouts in `matrix.zig`
- Dense (2D array)
- Symmetric
- Upper Triangular
- Lower Triangular
- Diagonal
- Upper Bidiagonal
- Lower Bidiagonal
- Tridiagonal

### Unsupported Matrix Layouts in `matrix.zig`
- Banded
- Sparse

The reason banded and sparse layouts are not supported is because they require more information than rows and columns to
fully describe the matrix memory layout, making the goal of using minimum space to store a generic matrix much more complicated.
CBLAS also doesn't support sparse matrices.

Sparse matrices violate an assumption which works for all non-sparse matrices:
matrices with the same compile-time-known layout and run-time-known size have the same memory layout.
So, it wouldn't make sense to include them in this implementation.

## Using `blas.zig`

`blas.zig` depends on two system files:
1. `cblas.h` header file
2. CBLAS implementation library
    - e.g. ATLAS, OpenBLAS, etc...

ATLAS and OpenBLAS are often available as system libraries, but installing these libraries is beyond the scope of this guide.
To link a system library, add either of these lines to `build.zig`:

```zig
    module.linkSystemLibrary("name_of_library", .{});
    // or
    exe.linkSystemLibrary("name_of_library");
```

### Running Unit Tests

After installing a CBLAS implementation, unit tests can be run from the root directory with

```sh
zig build test
```

## Using `matrix.zig`

All examples are run assuming that `matrix.zig` is imported under the namespace `M`:
```zig
const M = @import("matrix.zig");
```

Defining a Matrix type:
```zig
const Mat4x3D = M.Matrix(f64, 4, 3, .{});
```

Defining a Vector type:
```zig
const Vec10F = M.Vector(f32, 10);
```

Some common matrix and vector types are already defined, using the following convention:

Matrix/Vector types:
- `Mat[N]S`: N x N fixed-size matrix storing scalar type S
- `Vec[N]S`: N-dimensional column vector storing scalar type S
- `MatS`: Dynamic-sized matrix storing scalar type S
- `VecS`: Dynamic-sized column vector storing scalar type S
Scalar type suffixes (for `S`):
- `F`: f32
- `D`: f64
- `CF`: std.math.Complex(f32)
- `CD`: std.math.Complex(f64)

Initializing a matrix:
```zig
var A = M.Mat4D{}; // Fixed-size matrices are stored on the stack storing 0.

var B = try M.MatD.init(allocator, 4, 4); // Dynamic matrices store on the heap, and init fills with 0.
defer B.deinit(allocator); // Always clean up!
var C = M.Mat4D.init(null, 4, 4); // Fixed-size matrices can init without an allocator (equivalent to .{})

// Initializes a 2x2 matrix using manually-input values
var D = M.MatD.init_tuples(allocator, .{
    .{1, 0},
    .{0, 1},
});

// Other Mat*.init_* functions are available. See comments for more info.

A.assign(C); // Sets all values in A to C. A and C are the exact same type.

A.assign_any(D) // Sets all values in A to D, leaving extra values as 0.
// assign_any works with any matrix type, but is slower since it can only operate per-element, not in contiguous chunks of memory.
```

Accessing elements:
```zig
var A = M.Mat2D{};
const topleft: f64 = A.get(0, 0);
const bottomright: f64 = A.get(1, 1);
A.set(1, 0, 42.0);

try std.testing.expectEqual(0.0, A.get(2, 2)); // out of bounds elements are zero
A.set(2, 2, 2.0); // writing out of bounds does nothing

const index: usize = A.flat_index(0, 1); // gets the index into the matrix's data slice for an element
const topright: f64 = A.get_data()[index];
const rowcol = A.unflat_index(index); // reverses the operation
try std.testing.expectEqual(.{0, 1}, rowcol);
// NOTE: unflat_index is slow for Triangular layouts, since it requires an integer sqrt to unflatten the index
```

Matrix-Matrix elementwise operations:
```zig
var A = M.Mat2D{};
var B = M.Mat2D{};
var result = M.Mat2D{};

// Matrices of the exact same type can perform elementwise operations, like add, sub, mul, div, pow
result.add(A, B);
// All binary operations between matrices are in the form:
// result.operation(Left, Right)

// Because operations are elementwise, it is safe to have a matrix be a result and an operand:
A.add(A, B); // A = A + B

var C = try M.MatD.init_tuples(allocator, .{
    .{1, 2},
    .{3, 4}
});
defer C.deinit(allocator);
// Operations with "any" in the name operate on any matrix type.
// The same rules and tradeoffs from "assign_any" apply here.
result.add_any(A, C);

var D = try M.MatD.init(allocator, 3, 3);
// C.add(C, D); // This will PANIC! Dynamic matrices must have the same runtime size when using non-"any" operations.
```

Matrix slicing:
```zig
var A = try M.MatD.init(allocator, 4, 4);
defer A.deinit(allocator);
var A_middle = A.slice(.{
    // inclusive
    .start_row = 1,
    .start_col = 1,
    // exclusive
    .end_row = 3,
    .end_col = 3,
});
// Slices can be used in matrix functions taking anytype
var B = try M.MatD.init_copy_any_matrix(allocator, A_middle); // 2x2 dynamic matrix
defer B.deinit(allocator);
B.add_any(B, A_middle);

// start_* defaults to 0,0
var A_topleft = A.slice(.{
    .end_row = 2,
    .end_col = 2,
});

// end_* defaults to the matrix size
var A_bottomright = A.slice(.{
    .start_row = 2,
    .start_col = 2,
});

// Slices do not implement operations themselves, so they can't be used as a result:
// Not possible: A_topleft.add(A_middle, A_bottomright);

// To assign to a section of a matrix:
A.assign_range_any_matrix(.{
    .start_row = 1,
    .start_col = 1,
    .end_row = 3,
    .end_col = 3,
}, B);
```
Slices are intended for conveniently copying small sections of matrices to others.
Because slices go through two levels of indirection, they are quite slow (slice -> matrix -> actual data).
