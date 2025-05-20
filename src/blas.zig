//!  Basic Linear Algebra Subprograms (BLAS) Wrapper for Zig
//! 
//! Conventions
//! - BLAS operation names are used, but scalar prefix is replaced with a generic type argument.
//! - Destination matrix is size (m x n) or (n x n).
//! 
//! Defaults
//! - Order: column-major
//! - Trans: none
//! - Diag: non-unit
//! - Leading dimension (ld*) defaults to 0. Any ld* value <= 0 is instead computed assuming the matrix is a 2D array of given size and layout.

const std = @import("std");
const CC = @cImport({
    @cInclude("cblas.h");
});
pub const z32 = std.math.Complex(f32);
pub const z64 = std.math.Complex(f64);

/// Memory ordering of matrix data
pub const Order = enum(c_uint) {
    ColMajor = CC.CblasColMajor,
    RowMajor = CC.CblasRowMajor,
};

/// Whether to use an upper or lower triangular view of a 2D array
pub const UpLo = enum(c_uint) {
    Upper = CC.CblasUpper,
    Lower = CC.CblasLower,
};

/// Whether to place matrix A to the left or right of matrix B
pub const Side = enum(c_uint) {
    Left = CC.CblasLeft,
    Right = CC.CblasRight,
};

/// Whether to transpose a matrix in the function. Effect depends on context.
pub const Trans = enum(c_uint) {
    None = CC.CblasNoTrans,
    Transpose = CC.CblasTrans,
    ConjTrans = CC.CblasConjTrans,
    NoConjTrans = CC.CblasConjNoTrans,
};

/// Whether the diagonal of a triangular matrix is all 1's, or not.
pub const Diag = enum(c_uint) {
    NonUnit = CC.CblasNonUnit,
    Unit = CC.CblasUnit,
};

/// The leading dimension for a matrix layed out as a 2D array with no padding.
pub inline fn ld(rows: c_int, cols: c_int, order: Order, trans: Trans) c_int {
    return switch (trans) {
        .None, .NoConjTrans => switch (order) { .ColMajor => rows, .RowMajor => cols },
        .Transpose, .ConjTrans => switch (order) { .ColMajor => cols, .RowMajor => rows },
    };
}

/// The leading dimension for a square matrix layed out as a 2D array with no padding.
/// `rows` and `cols` are the matrix adjacent to this one, depending on side:
/// - Left: A(rows x rows) * B(rows x cols)
/// - Right: B(rows x cols) * A(cols x cols)
pub inline fn ld_square(rows: c_int, cols: c_int, side: Side) c_int {
    return switch (side) {
        .Left => rows,
        .Right => cols,
    };
}

/// Gets the base type of a scalar.
/// For floats, returns S.
/// For complex floats, returns the type of the real and imaginary components.
pub fn BaseType(comptime S: type) type {
    return switch (comptime S) {
        f32 => f32,
        f64 => f64,
        z32 => f32,
        z64 => f64,
        else => @compileError("unsupported scalar type")
    };
}

pub const OptionsV = struct {
    incx: c_int = 1,
};

pub const OptionsVV = struct {
    incx: c_int = 1,
    incy: c_int = 1,
};

/// Adds two vectors, scaling one of them.
/// 
/// `y = alpha*x + y`
pub fn axpy(comptime S: type, n: usize, alpha: S, x: [*]const S, y: [*]S, options: OptionsVV) void {
    const cn: c_int = @intCast(n);
    switch (comptime S) {
        f32 => CC.cblas_saxpy(cn, alpha, x, options.incx, y, options.incy),
        f64 => CC.cblas_daxpy(cn, alpha, x, options.incx, y, options.incy),
        z32 => CC.cblas_caxpy(cn, &alpha, x, options.incx, y, options.incy),
        z64 => CC.cblas_zaxpy(cn, &alpha, x, options.incx, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Adds two vectors, scaling both of them.
///
/// `y = alpha*x + beta*y`
pub fn axpby(comptime S: type, n: usize, alpha: S, x: [*]const S, beta: S, y: [*]S, options: OptionsVV) void {
    const cn: c_int = @intCast(n);
    switch (comptime S) {
        f32 => CC.cblas_saxpby(cn, alpha, x, options.incx, beta, y, options.incy),
        f64 => CC.cblas_daxpby(cn, alpha, x, options.incx, beta, y, options.incy),
        z32 => CC.cblas_caxpby(cn, &alpha, x, options.incx, beta, y, options.incy),
        z64 => CC.cblas_zaxpby(cn, &alpha, x, options.incx, beta, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Scales a vector.
///
/// `x = alpha*x`
pub fn scal(comptime S: type, n: usize, alpha: S, x: [*]S, options: OptionsV) void {
    const cn: c_int = @intCast(n);
    switch (comptime S) {
        f32 => CC.cblas_sscal(cn, alpha, x, options.incx),
        f64 => CC.cblas_dscal(cn, alpha, x, options.incx),
        z32 => CC.cblas_cscal(cn, &alpha, x, options.incx),
        z64 => CC.cblas_zscal(cn, &alpha, x, options.incx),
        else => @compileError("unsupported scalar type")
    }
}

/// Swaps the contents of two vectors.
///
/// `x, y = y, x`
pub fn swap(comptime S: type, n: usize, x: [*]S, y: [*]S, options: OptionsVV) void {
    const cn: c_int = @intCast(n);
    switch (comptime S) {
        f32 => CC.cblas_sswap(cn, x, options.incx, y, options.incy),
        f64 => CC.cblas_dswap(cn, x, options.incx, y, options.incy),
        z32 => CC.cblas_cswap(cn, x, options.incx, y, options.incy),
        z64 => CC.cblas_zswap(cn, x, options.incx, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Computes the dot (inner) product between two vectors.
///
/// `x^T*y`
pub fn dot(comptime S: type, n: usize, x: [*]const S, y: [*]const S, options: OptionsVV) S {
    const cn: c_int = @intCast(n);
    switch (comptime S) {
        f32 => return CC.cblas_sdot(cn, x, options.incx, y, options.incy),
        f64 => return CC.cblas_ddot(cn, x, options.incx, y, options.incy),
        z32 => {
            var out: S = undefined;
            CC.cblas_cdotu_sub(cn, x, options.incx, y, options.incy, &out);
            return out;
        },
        z64 => {
            var out: S = undefined;
            CC.cblas_zdotu_sub(cn, x, options.incx, y, options.incy, &out);
            return out;
        },
        else => @compileError("unsupported scalar type")
    }
}

/// Computes the complex conjugate dot (inner) product
///
/// `x^H*y`
pub fn dotc(comptime S: type, n: usize, x: [*]const S, y: [*]const S, options: OptionsVV) S {
    const cn: c_int = @intCast(n);
    var out: S = undefined;
    switch (comptime S) {
        z32 => CC.cblas_cdotc_sub(cn, x, options.incx, y, options.incy, &out),
        z64 => CC.cblas_zdotc_sub(cn, x, options.incx, y, options.incy, &out),
        else => @compileError("unsupported scalar type")
    }
    return out;
}

/// Computes the euclidean norm, length of the vector in space.
///
/// `||x||`
pub fn nrm2(comptime S: type, n: usize, x: [*]const S, options: OptionsV) BaseType(S) {
    const cn: c_int = @intCast(n);
    return switch (comptime S) {
        f32 => CC.cblas_snrm2(cn, x, options.incx),
        f64 => CC.cblas_dnrm2(cn, x, options.incx),
        z32 => CC.cblas_scnrm2(cn, x, options.incx),
        z64 => CC.cblas_dznrm2(cn, x, options.incx),
        else => @compileError("unsupported scalar type")
    };
}

/// Computes the 1 norm, aka. taxicab norm, sum of absolute values of components.
///
/// `|x|_1`
pub fn asum(comptime S: type, n: usize, x: [*]const S, options: OptionsV) BaseType(S) {
    const cn: c_int = @intCast(n);
    return switch (comptime S) {
        f32 => CC.cblas_sasum(cn, x, options.incx),
        f64 => CC.cblas_dasum(cn, x, options.incx),
        z32 => CC.cblas_scasum(cn, x, options.incx),
        z64 => CC.cblas_dzasum(cn, x, options.incx),
        else => @compileError("unsupported scalar type")
    };
}

/// Computes the Index of Absolute Maximum in the vector, the index of the largest element by absolute value.
/// 
/// `argmax(x)`
pub fn iamax(comptime S: type, n: usize, x: [*]const S, options: OptionsV) usize {
    const cn: c_int = @intCast(n);
    return switch (comptime S) {
        f32 => CC.cblas_isamax(cn, x, options.incx),
        f64 => CC.cblas_idamax(cn, x, options.incx),
        z32 => CC.cblas_icamax(cn, x, options.incx),
        z64 => CC.cblas_izamax(cn, x, options.incx),
        else => @compileError("unsupported scalar type")
    };
}

/// Computes the Index of Absolute Minimum in the vector, the index of the smallest element by absolute value.
/// 
/// `argmin(x)`
pub fn iamin(comptime S: type, n: usize, x: [*]const S, options: OptionsV) usize {
    const cn: c_int = @intCast(n);
    return switch (comptime S) {
        f32 => CC.cblas_isamin(cn, x, options.incx),
        f64 => CC.cblas_idamin(cn, x, options.incx),
        z32 => CC.cblas_icamin(cn, x, options.incx),
        z64 => CC.cblas_izamin(cn, x, options.incx),
        else => @compileError("unsupported scalar type")
    };
}

/// Generates a 2D rotation matrix `R=[c, s; -s, c]` such that
/// `R*[a; b] = [r; 0]`.
/// - `c = R(0,0)`
/// - `s = R(0,1)`
/// - `a = r`
/// - `b =` an auxilary value which can be used in subsequent calls
pub fn rotg(comptime S: type, a: *S, b: *S, c: *BaseType(S), s: *S) void {
    switch (comptime S) {
        f32 => CC.cblas_srotg(&a, &b, &c, &s),
        f64 => CC.cblas_drotg(&a, &b, &c, &s),
        z32 => CC.cblas_crotg(&a, &b, &c, &s),
        z64 => CC.cblas_zrotg(&a, &b, &c, &s),
        else => @compileError("unsupported scalar type")
    }
}

/// Applies a 2D rotation matrix `R=[c s; -s c]` to each pair of elements in vectors x and y:
/// 
/// `[x_i; y_i] = R*[x_i; y_i]`
pub fn rot(comptime S: type, n: usize, x: [*]S, y: [*]S, c: BaseType(S), s: BaseType(S), options: OptionsVV) void {
    const cn: c_int = @intCast(n);
    switch (comptime S) {
        f32 => CC.cblas_srot(cn, x, options.incx, y, options.incy, c, s),
        f64 => CC.cblas_drot(cn, x, options.incx, y, options.incy, c, s),
        z32 => CC.cblas_csrot(n, x, options.incx, y, options.incy, c, s),
        z64 => CC.cblas_zdrot(n, x, options.incx, y, options.incy, c, s),
        else => @compileError("unsupported scalar type")
    }
}

/// Generates 2x2 matrix H such that
/// `H * [sqrt(d1), b1; sqrt(d2), b2] = [r; 0]`.
/// `param` is a representation of H which is passed to `rotm`.
/// `param[0]` determines the usage of other param entries, which are in column major format of `h[1..5]`:
/// - `-2.0`: 2x2 identity matrix `[1, 0; 0, 1]`
/// - `-1.0`: 2x2 matrix of form `[param[1], param[3]; param[2], param[4]]`
/// - `0.0`: 2x2 matrix of form `[param[1], 0; 0, param[4]]`
/// - `1.0`: 2x2 matrix of form `[0, param[3]; param[2], 0]`
pub fn rotmg(comptime S: type, d1: *S, d2: *S, b1: *S, b2: S) [5]S {
    var param: [5]S = undefined;
    switch (comptime S) {
        f32 => CC.cblas_srotmg(d1, d2, b1, b2, &param),
        f64 => CC.cblas_drotmg(d1, d2, b1, b2, &param),
        else => @compileError("unsupported scalar type")
    }
}

/// Applies a 2D transformation H encoded in param (see `rotmg`) to each pair of elements in vectors x and y:
/// 
/// `[x_i; y_i] = H*[x_i; y_i]`
pub fn rotm(comptime S: type, n: usize, x: [*]S, y: [*]S, param: [5]S, options: OptionsVV) void {
    const cn: c_int = @intCast(n);
    switch (comptime S) {
        f32 => CC.cblas_srotm(cn, x, options.incx, y, options.incy, &param),
        f64 => CC.cblas_drotm(cn, x, options.incx, y, options.incy, &param),
        else => @compileError("unsupported scalar type")
    }
}

// Level 2 BLAS

pub const OptionsMV = struct {
    order: Order = .ColMajor,
    lda: c_int = 0,
    incx: c_int = 1,
};

pub const OptionsMVV = struct {
    order: Order = .ColMajor,
    lda: c_int = 0,
    incx: c_int = 1,
    incy: c_int = 1,
};

pub const OptionsTMVV = struct {
    order: Order = .ColMajor,
    trans: Trans = .None,
    lda: c_int = 0,
    incx: c_int = 1,
    incy: c_int = 1,
};

pub const OptionsTrV = struct {
    order: Order = .ColMajor,
    trans: Trans = .None,
    diag: Diag = .NonUnit,
    lda: c_int = 0,
    incx: c_int = 1,
};

pub const OptionsTpV = struct {
    order: Order = .ColMajor,
    trans: Trans = .None,
    diag: Diag = .NonUnit,
    incx: c_int = 1,
};

pub const OptionsPV = struct {
    order: Order = .ColMajor,
    incx: c_int = 1,
};

pub const OptionsPVV = struct {
    order: Order = .ColMajor,
    incx: c_int = 1,
    incy: c_int = 1,
};

/// General Matrix-Vector multiply.
///
/// `y = alpha*A*x + beta*y`
pub fn gemv(comptime S: type, m: c_int, n: c_int, alpha: S, A: [*]const S, x: [*]const S, beta: S, y: [*]S, options: OptionsTMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const trans: c_uint = @intFromEnum(options.trans);
    const lda: c_int = if (options.lda > 0) options.lda else ld(m, n, options.order, options.trans);
    switch (comptime S) {
        f32 => CC.cblas_sgemv(order, trans, m, n, alpha, A, lda, x, options.incx, beta, y, options.incy),
        f64 => CC.cblas_dgemv(order, trans, m, n, alpha, A, lda, x, options.incx, beta, y, options.incy),
        z32 => CC.cblas_cgemv(order, trans, m, n, &alpha, A, lda, x, options.incx, &beta, y, options.incy),
        z64 => CC.cblas_zgemv(order, trans, m, n, &alpha, A, lda, x, options.incx, &beta, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Hermitian Matrix-Vector multiply.
///
/// `y = alpha*A*x + beta*y`, where A is a Hermitian matrix.
pub fn hemv(comptime S: type, uplo: UpLo, n: c_int, alpha: S, A: [*]const S, x: [*]const S, beta: S, y: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        z32 => CC.cblas_chemv(order, cuplo, n, &alpha, A, lda, x, options.incx, &beta, y, options.incy),
        z64 => CC.cblas_zhemv(order, cuplo, n, &alpha, A, lda, x, options.incx, &beta, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Symmetric Matrix-Vector multiply.
///
/// `y = alpha*A*x + beta*y`, where A is a symmetric matrix.
pub fn symv(comptime S: type, uplo: UpLo, n: c_int, alpha: S, A: [*]const S, x: [*]const S, beta: S, y: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        f32 => CC.cblas_ssymv(order, cuplo, n, alpha, A, lda, x, options.incx, beta, y, options.incy),
        f64 => CC.cblas_dsymv(order, cuplo, n, alpha, A, lda, x, options.incx, beta, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Triangular Matrix-Vector multiply.
/// 
/// `x = A*x`, where A is a triangular matrix.
pub fn trmv(comptime S: type, uplo: UpLo, n: c_int, A: [*]const S, x: [*]S, options: OptionsTrV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    const trans: c_uint = @intFromEnum(options.trans);
    const diag: c_uint = @intFromEnum(options.diag);
    switch (comptime S) {
        f32 => CC.cblas_strmv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        f64 => CC.cblas_dtrmv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        z32 => CC.cblas_ctrmv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        z64 => CC.cblas_ztrmv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        else => @compileError("unsupported scalar type")
    }
}

/// Triangular Solve.
/// 
/// `x = A^-1*x`, where A is a triangular matrix.
pub fn trsv(comptime S: type, uplo: UpLo, n: c_int, A: [*]const S, x: [*]S, options: OptionsTrV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    const trans: c_uint = @intFromEnum(options.trans);
    const diag: c_uint = @intFromEnum(options.diag);
    switch (comptime S) {
        f32 => CC.cblas_strsv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        f64 => CC.cblas_dtrsv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        z32 => CC.cblas_ctrsv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        z64 => CC.cblas_ztrsv(order, cuplo, trans, diag, n, A, lda, x, options.incx),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Hermitian Matrix-Vector multiply. Packed layout is upper or lower triangular.
///
/// `y = alpha*A*x + beta*y`, where A is a Hermitian matrix.
pub fn hpmv(comptime S: type, uplo: UpLo, n: c_int, alpha: S, A: [*]const S, x: [*]const S, beta: S, y: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        z32 => CC.cblas_chpmv(order, cuplo, n, &alpha, A, lda, x, options.incx, &beta, y, options.incy),
        z64 => CC.cblas_zhemv(order, cuplo, n, &alpha, A, lda, x, options.incx, &beta, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Symmetric Matrix-Vector multiply. Packed layout is upper or lower triangular.
///
/// `y = alpha*A*x + beta*y`, where A is a symmetric matrix.
pub fn spmv(comptime S: type, uplo: UpLo, n: c_int, alpha: S, A: [*]const S, x: [*]const S, beta: S, y: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    switch (comptime S) {
        f32 => CC.cblas_sspmv(order, cuplo, n, alpha, A, x, options.incx, beta, y, options.incy),
        f64 => CC.cblas_dspmv(order, cuplo, n, alpha, A, x, options.incx, beta, y, options.incy),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Triangular Matrix-Vector multiply. Packed layout is upper or lower triangular.
/// 
/// `x = A*x`, where A is a triangular matrix.
pub fn tpmv(comptime S: type, uplo: UpLo, n: c_int, A: [*]const S, x: [*]S, options: OptionsTpV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.trans);
    const diag: c_uint = @intFromEnum(options.diag);
    switch (comptime S) {
        f32 => CC.cblas_stpmv(order, cuplo, trans, diag, n, A, x, options.incx),
        f64 => CC.cblas_dtpmv(order, cuplo, trans, diag, n, A, x, options.incx),
        z32 => CC.cblas_ctpmv(order, cuplo, trans, diag, n, A, x, options.incx),
        z64 => CC.cblas_ztpmv(order, cuplo, trans, diag, n, A, x, options.incx),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Triangular Solve. Packed layout is upper or lower triangular.
/// 
/// `x = A^-1*x`, where A is a triangular matrix.
pub fn tpsv(comptime S: type, uplo: UpLo, n: c_int, A: [*]const S, x: [*]S, options: OptionsTpV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.trans);
    const diag: c_uint = @intFromEnum(options.diag);
    switch (comptime S) {
        f32 => CC.cblas_stpsv(order, cuplo, trans, diag, n, A, x, options.incx),
        f64 => CC.cblas_dtpsv(order, cuplo, trans, diag, n, A, x, options.incx),
        z32 => CC.cblas_ctpsv(order, cuplo, trans, diag, n, A, x, options.incx),
        z64 => CC.cblas_ztpsv(order, cuplo, trans, diag, n, A, x, options.incx),
        else => @compileError("unsupported scalar type")
    }
}

/// General Rank-1 update.
/// 
/// `A = A + alpha*x*y^T`
pub fn ger(comptime S: type, m: c_int, n: c_int, alpha: S, x: [*]const S, y: [*]const S, A: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        f32 => CC.cblas_sger(order, m, n, alpha, x, options.incx, y, options.incy, A, lda),
        f64 => CC.cblas_dger(order, m, n, alpha, x, options.incx, y, options.incy, A, lda),
        z32 => CC.cblas_cgeru(order, m, n, &alpha, x, options.incx, y, options.incy, A, lda),
        z64 => CC.cblas_zgeru(order, m, n, &alpha, x, options.incy, y, options.incy, A, lda),
        else => @compileError("unsupported scalar type")
    }
}

/// General Rank-1 update, Complex conjugate transpose.
/// 
/// `A = A + alpha*x*y^H`
pub fn gerc(comptime S: type, m: c_int, n: c_int, alpha: S, x: [*]const S, y: [*]const S, A: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        z32 => CC.cblas_cgerc(order, m, n, &alpha, x, options.incx, y, options.incy, A, lda),
        z64 => CC.cblas_zgerc(order, m, n, &alpha, x, options.incy, y, options.incy, A, lda),
        else => @compileError("unsupported scalar type")
    }
}

/// Symmetric Rank-1 update.
/// 
/// `A = A + alpha*x*x^T`
pub fn syr(comptime S: type, uplo: UpLo, n: c_int, alpha: S, x: [*]const S, A: [*]S, options: OptionsMV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        f32 => CC.cblas_ssyr(order, cuplo, n, alpha, x, options.incx, A, lda),
        f64 => CC.cblas_dsyr(order, cuplo, n, alpha, x, options.incx, A, lda),
        else => @compileError("unsupported scalar type")
    }
}

/// Hermitian Rank-1 update.
/// 
/// `A = A + alpha*x*x^H`
pub fn her(comptime S: type, uplo: UpLo, n: c_int, alpha: S, x: [*]const S, A: [*]S, options: OptionsMV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        z32 => CC.cblas_cher(order, cuplo, n, &alpha, x, options.incx, A, lda),
        z64 => CC.cblas_zher(order, cuplo, n, &alpha, x, options.incx, A, lda),
        else => @compileError("unsupported scalar type")
    }
}

/// Symmetric Rank-2 update.
/// 
/// `A = A + alpha*x*y^T + alpha*y*x^T`
pub fn syr2(comptime S: type, uplo: UpLo, n: c_int, alpha: S, x: [*]const S, y: [*]const S, A: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        f32 => CC.cblas_ssyr2(order, cuplo, n, alpha, x, options.incx, y, options.incy, A, lda),
        f64 => CC.cblas_dsyr2(order, cuplo, n, alpha, x, options.incx, y, options.incy, A, lda),
        else => @compileError("unsupported scalar type")
    }
}

/// Hermitian Rank-2 update.
/// 
/// `A = A + alpha*x*y^H + alpha*y*x^H`
pub fn her2(comptime S: type, uplo: UpLo, n: c_int, alpha: S, x: [*]const S, y: [*]const S, A: [*]S, options: OptionsMVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else n;
    switch (comptime S) {
        z32 => CC.cblas_cher2(order, cuplo, n, &alpha, x, options.incx, y, options.incy, A, lda),
        z64 => CC.cblas_zher2(order, cuplo, n, &alpha, x, options.incx, y, options.incy, A, lda),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Symmetric Rank-1 update. Packed layout is upper or lower triangular.
/// 
/// `A = A + alpha*x*x^T`
pub fn spr(comptime S: type, uplo: UpLo, n: c_int, alpha: S, x: [*]const S, A: [*]S, options: OptionsPV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    switch (comptime S) {
        f32 => CC.cblas_sspr(order, cuplo, n, alpha, x, options.incx, A),
        f64 => CC.cblas_dspr(order, cuplo, n, alpha, x, options.incx, A),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Hermitian Rank-1 update. Packed layout is upper or lower triangular.
/// 
/// `A = A + alpha*x*x^H`
pub fn hpr(comptime S: type, uplo: UpLo, n: c_int, alpha: BaseType(S), x: [*]const S, A: [*]S, options: OptionsPV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    switch (comptime S) {
        z32 => CC.cblas_chpr(order, cuplo, n, alpha, x, options.incx, A),
        z64 => CC.cblas_zhpr(order, cuplo, n, alpha, x, options.incx, A),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Symmetric Rank-2 update. Packed layout is upper or lower triangular.
/// 
/// `A = A + alpha*x*y^T + alpha*y*x^T`
pub fn spr2(comptime S: type, uplo: UpLo, n: c_int, alpha: S, x: [*]const S, y: [*]const S, A: [*]S, options: OptionsPVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    switch (comptime S) {
        f32 => CC.cblas_sspr2(order, cuplo, n, alpha, x, options.incx, y, options.incy, A),
        f64 => CC.cblas_dspr2(order, cuplo, n, alpha, x, options.incx, y, options.incy, A),
        else => @compileError("unsupported scalar type")
    }
}

/// Packed Hermitian Rank-2 update. Packed layout is upper or lower triangular.
/// 
/// `A = A + alpha*x*y^H + alpha*y*x^H`
pub fn hpr2(comptime S: type, uplo: UpLo, n: c_int, alpha: S, x: [*]const S, y: [*]const S, A: [*]S, options: OptionsPVV) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    switch (comptime S) {
        z32 => CC.cblas_chpr2(order, cuplo, n, &alpha, x, options.incx, y, options.incy, A),
        z64 => CC.cblas_zhpr2(order, cuplo, n, &alpha, x, options.incx, y, options.incy, A),
        else => @compileError("unsupported scalar type")
    }
}

// Level 3 BLAS

const OptionsTTMMM = struct {
    order: Order = .ColMajor,
    transA: Trans = .None,
    transB: Trans = .None,
    lda: c_int = 0,
    ldb: c_int = 0,
    ldc: c_int = 0,
};

const OptionsMMM = struct {
    order: Order = .ColMajor,
    lda: c_int = 0,
    ldb: c_int = 0,
    ldc: c_int = 0,
};

const OptionsTrMM = struct {
    order: Order = .ColMajor,
    transA: Trans = .None,
    diag: Diag = .NonUnit,
    lda: c_int = 0,
    ldb: c_int = 0,
};

pub const OptionsTM_M = struct {
    order: Order = .ColMajor,
    trans: Trans = .None,
    lda: c_int = 0,
    ldc: c_int = 0,
};

pub const OptionsTMMM = struct {
    order: Order = .ColMajor,
    trans: Trans = .None,
    lda: c_int = 0,
    ldb: c_int = 0,
    ldc: c_int = 0,
};

/// General Matrix-Matrix multiplication.
/// 
/// `C = alpha*A*B + beta*C`
pub fn gemm(comptime S: type, m: c_int, n: c_int, k: c_int, alpha: S, A: [*]const S, B: [*]const S, beta: S, C: [*]S, options: OptionsTTMMM) void {
    const order: c_uint = @intFromEnum(options.order);
    const transA: c_uint = @intFromEnum(options.transA);
    const transB: c_uint = @intFromEnum(options.transB);
    const lda: c_int = if (options.lda > 0) options.lda else ld(m, k, options.order, options.transA);
    const ldb: c_int = if (options.ldb > 0) options.ldb else ld(k, n, options.order, options.transB);
    const ldc: c_int = if (options.ldc > 0) options.ldc else ld(m, n, options.order, .None);
    switch (comptime S) {
        f32 => CC.cblas_sgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc),
        f64 => CC.cblas_dgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc),
        z32 => CC.cblas_cgemm(order, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc),
        z64 => CC.cblas_zgemm(order, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc),
        else => @compileError("unsupported scalar type")
    }
}

/// Symmetric Matrix-Matrix multiplication.
/// 
/// `C = alpha*A*B + beta*C`
pub fn symm(comptime S: type, side: Side, uplo: UpLo, m: c_int, n: c_int, alpha: S, A: [*]const S, B: [*]const S, beta: S, C: [*]S, options: OptionsMMM) void {
    const order: c_uint = @intFromEnum(options.order);
    const cside: c_uint = @intFromEnum(side);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else ld_square(m, n, side);
    const ldb: c_int = if (options.ldb > 0) options.ldb else ld(m, n, options.order, .None);
    const ldc: c_int = if (options.ldc > 0) options.ldc else ld(m, n, options.order, .None);
    switch (comptime S) {
        f32 => CC.cblas_ssymm(order, cside, cuplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc),
        f64 => CC.cblas_dsymm(order, cside, cuplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc),
        z32 => CC.cblas_csymm(order, cside, cuplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc),
        z64 => CC.cblas_zsymm(order, cside, cuplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc),
        else => @compileError("unsupported scalar type")
    }
}

/// Hermitian Matrix-Matrix multiplication.
/// 
/// `C = alpha*A*B + beta*C`
pub fn hemm(comptime S: type, side: Side, uplo: UpLo, m: c_int, n: c_int, alpha: S, A: [*]const S, B: [*]const S, beta: S, C: [*]S, options: OptionsMMM) void {
    const order: c_uint = @intFromEnum(options.order);
    const cside: c_uint = @intFromEnum(side);
    const cuplo: c_uint = @intFromEnum(uplo);
    const lda: c_int = if (options.lda > 0) options.lda else ld_square(m, n, side);
    const ldb: c_int = if (options.ldb > 0) options.ldb else ld(m, n, options.order, .None);
    const ldc: c_int = if (options.ldc > 0) options.ldc else ld(m, n, options.order, .None);
    switch (comptime S) {
        z32 => CC.cblas_chemm(order, cside, cuplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc),
        z64 => CC.cblas_zhemm(order, cside, cuplo, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc),
        else => @compileError("unsupported scalar type")
    }
}

/// Triangular Matrix-Matrix multiplication.
/// 
/// - Side left: `B = alpha*A*B`
/// - Side right: `B = alpha*B*A`
pub fn trmm(comptime S: type, side: Side, uplo: UpLo, m: c_int, n: c_int, alpha: S, A: [*]const S, B: [*]S, options: OptionsTrMM) void {
    const order: c_uint = @intFromEnum(options.order);
    const cside: c_uint = @intFromEnum(side);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.transA);
    const diag: c_uint = @intFromEnum(options.diag);
    const lda: c_int = if (options.lda > 0) options.lda else ld_square(m, n, side);
    const ldb: c_int = if (options.ldb > 0) options.ldb else ld(m, n, options.order, .None);
    switch (comptime S) {
        f32 => CC.cblas_strmm(order, cside, cuplo, trans, diag, m, n, alpha, A, lda, B, ldb),
        f64 => CC.cblas_dtrmm(order, cside, cuplo, trans, diag, m, n, alpha, A, lda, B, ldb),
        z32 => CC.cblas_ctrmm(order, cside, cuplo, trans, diag, m, n, &alpha, A, lda, B, ldb),
        z64 => CC.cblas_ztrmm(order, cside, cuplo, trans, diag, m, n, &alpha, A, lda, B, ldb),
        else => @compileError("unsupported scalar type")
    }
}

/// Triangular Solve Matrix.
/// 
/// - Side left: `B = alpha*A^-1*B`
/// - Side right: `B = alpha*B*A^-1`
pub fn trsm(comptime S: type, side: Side, uplo: UpLo, m: c_int, n: c_int, alpha: S, A: [*]const S, B: [*]S, options: OptionsTrMM) void {
    const order: c_uint = @intFromEnum(options.order);
    const cside: c_uint = @intFromEnum(side);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.transA);
    const diag: c_uint = @intFromEnum(options.diag);
    const lda: c_int = if (options.lda > 0) options.lda else ld_square(m, n, side);
    const ldb: c_int = if (options.ldb > 0) options.ldb else ld(m, n, options.order, .None);
    switch (comptime S) {
        f32 => CC.cblas_strsm(order, cside, cuplo, trans, diag, m, n, alpha, A, lda, B, ldb),
        f64 => CC.cblas_dtrsm(order, cside, cuplo, trans, diag, m, n, alpha, A, lda, B, ldb),
        z32 => CC.cblas_ctrsm(order, cside, cuplo, trans, diag, m, n, &alpha, A, lda, B, ldb),
        z64 => CC.cblas_ztrsm(order, cside, cuplo, trans, diag, m, n, &alpha, A, lda, B, ldb),
        else => @compileError("unsupported scalar type")
    }
}

/// Symmetric Rank-K update.
/// 
/// `C = alpha*A*A^T + beta*C`
pub fn syrk(comptime S: type, uplo: UpLo, n: c_int, k: c_int, alpha: S, A: [*]const S, beta: S, C: [*]S, options: OptionsTM_M) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.trans);
    const lda: c_int = if (options.lda > 0) options.lda else ld(n, k, options.order, options.trans);
    const ldc: c_int = if (options.ldc > 0) options.ldc else n;
    switch (comptime S) {
        f32 => CC.cblas_ssyrk(order, cuplo, trans, n, k, alpha, A, lda, beta, C, ldc),
        f64 => CC.cblas_dsyrk(order, cuplo, trans, n, k, alpha, A, lda, beta, C, ldc),
        z32 => CC.cblas_csyrk(order, cuplo, trans, n, k, &alpha, A, lda, &beta, C, ldc),
        z64 => CC.cblas_zsyrk(order, cuplo, trans, n, k, &alpha, A, lda, &beta, C, ldc),
        else => @compileError("unsupported scalar type")
    }
}

/// Hermitian Rank-K update.
/// 
/// `C = alpha*A*A^H + beta*C`
pub fn herk(comptime S: type, uplo: UpLo, n: c_int, k: c_int, alpha: BaseType(S), A: [*]const S, beta: BaseType(S), C: [*]S, options: OptionsTM_M) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.trans);
    const lda: c_int = if (options.lda > 0) options.lda else ld(n, k, options.order, options.trans);
    const ldc: c_int = if (options.ldc > 0) options.ldc else n;
    switch (comptime S) {
        z32 => CC.cblas_cherk(order, cuplo, trans, n, k, alpha, A, lda, beta, C, ldc),
        z64 => CC.cblas_zherk(order, cuplo, trans, n, k, alpha, A, lda, beta, C, ldc),
        else => @compileError("unsupported scalar type")
    }
}

/// Symmetric Rank-2K update.
/// 
/// `C = alpha*A*B^T + conj(alpha)*B*A^T + beta*C`
pub fn syr2k(comptime S: type, uplo: UpLo, n: c_int, k: c_int, alpha: S, A: [*]const S, B: [*]const S, beta: S, C: [*]S, options: OptionsTMMM) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.trans);
    const lda: c_int = if (options.lda > 0) options.lda else ld(n, k, options.order, options.trans);
    const ldb: c_int = if (options.ldb > 0) options.ldb else ld(n, k, options.order, options.trans);
    const ldc: c_int = if (options.ldc > 0) options.ldc else n;
    switch (comptime S) {
        f32 => CC.cblas_ssyr2k(order, cuplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc),
        f64 => CC.cblas_dsyr2k(order, cuplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc),
        z32 => CC.cblas_csyr2k(order, cuplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc),
        z64 => CC.cblas_zsyr2k(order, cuplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc),
        else => @compileError("unsupported scalar type")
    }
}

/// Hermitian Rank-2K update.
/// 
/// `C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C`
pub fn her2k(comptime S: type, uplo: UpLo, n: c_int, k: c_int, alpha: S, A: [*]const S, B: [*]const S, beta: BaseType(S), C: [*]S, options: OptionsTMMM) void {
    const order: c_uint = @intFromEnum(options.order);
    const cuplo: c_uint = @intFromEnum(uplo);
    const trans: c_uint = @intFromEnum(options.trans);
    const lda: c_int = if (options.lda > 0) options.lda else ld(n, k, options.order, options.trans);
    const ldb: c_int = if (options.ldb > 0) options.ldb else ld(n, k, options.order, options.trans);
    const ldc: c_int = if (options.ldc > 0) options.ldc else n;
    switch (comptime S) {
        z32 => CC.cblas_cher2k(order, cuplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc),
        z64 => CC.cblas_zher2k(order, cuplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc),
        else => @compileError("unsupported scalar type")
    }
}