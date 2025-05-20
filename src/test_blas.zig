const std = @import("std");
const testing = std.testing;
const blas = @import("blas.zig");
const M = @import("matrix.zig");


test "level 1" {
    var x = M.Vec3D{};
    x.vector_assign_tuple(.{ 1, 2, 3 });
    var y = M.Vec3D{};
    y.vector_assign_tuple(.{ 3, 2, 1 });
    var e = M.Vec3D{};
    e.vector_assign_tuple(.{ 4, 4, 4 });
    blas.axpy(f64, 3, 1.0, x.get_ptr(), y.get_ptr(), .{});
    try testing.expectEqual(e.data, y.data);

    try testing.expectEqual(24.0, blas.dot(f64, 3, x.get_ptr(), y.get_ptr(), .{}));

    try testing.expectEqual(6, blas.asum(f64, 3, x.get_ptr(), .{}));
}

test "level 2" {
    var A = M.Mat3D{};
    A.assign_tuples(.{
        .{1, 0, 0},
        .{0, 2, 0},
        .{0, 0, 3}
    });
    var x = M.Vec3D{};
    x.vector_assign_tuple(.{3, 2, 1});
    var y = M.Vec3D{};
    var e = M.Vec3D{};
    e.vector_assign_tuple(.{3, 4, 3});
    blas.gemv(f64, 3, 3, 1.0, A.get_ptr(), x.get_ptr(), 0.0, y.get_ptr(), .{});
    try testing.expectEqual(e.data, y.data);

    blas.symv(f64, .Upper, 3, 1.0, A.get_ptr(), x.get_ptr(), 0.0, y.get_ptr(), .{});
    try testing.expectEqual(e.data, y.data);

    y.assign(x);
    blas.trmv(f64, .Upper, 3, A.get_ptr(), y.get_ptr(), .{});
    try testing.expectEqual(e.data, y.data);

    var B = M.Matrix(f64, 3, 3, .{.layout = .UpperTriangular}){};
    B.assign_any_matrix(A);

    y.assign(x);
    blas.tpmv(f64, .Upper, 3, B.get_ptr(), y.get_ptr(), .{});
    try testing.expectEqual(e.data, y.data);

    blas.spmv(f64, .Upper, 3, 1.0, B.get_ptr(), x.get_ptr(), 0.0, y.get_ptr(), .{});
    try testing.expectEqual(e.data, y.data);
}

test "level 3" {
    var A = M.Mat3D{};
    var B = M.Mat3D{};
    var C = M.Mat3D{};
    var E = M.Mat3D{};
    A.assign_tuples(.{
        .{1, 0, 0},
        .{0, 0, 1},
        .{0, 1, 0},
    });
    B.assign_tuples(.{
        .{1, 0, 0},
        .{0, 2, 0},
        .{0, 0, 3},
    });
    E.assign_tuples(.{
        .{1, 0, 0},
        .{0, 0, 3},
        .{0, 2, 0},
    });
    blas.gemm(f64, 3, 3, 3, 1.0, A.get_ptr(), B.get_ptr(), 0.0, C.get_ptr(), .{});
    try testing.expectEqual(E.data, C.data);

    C.fill(0);
    blas.symm(f64, .Left, .Upper, 3, 3, 1.0, A.get_ptr(), B.get_ptr(), 0.0, C.get_ptr(), .{});
    try testing.expectEqual(E.data, C.data);

    C.assign(A);
    blas.trmm(f64, .Right, .Upper, 3, 3, 1.0, B.get_ptr(), C.get_ptr(), .{});
    try testing.expectEqual(E.data, C.data);

    var y = M.Vec3D{};
    y.vector_assign_tuple(.{3, 4, 3});
    var x = M.Vec3D{};
    var e = M.Vec3D{};
    e.vector_assign_tuple(.{3, 2, 1});
    x.assign(y);
    blas.trsv(f64, .Upper, 3, B.get_ptr(), x.get_ptr(), .{});
    try testing.expectEqual(e.data, x.data);

    var D = M.Matrix(f64, 3, 3, .{.layout = .UpperTriangular}){};
    D.assign_any_matrix(B);
    x.assign(y);
    blas.tpsv(f64, .Upper, 3, D.get_ptr(), x.get_ptr(), .{});
    try testing.expectEqual(e.data, x.data);

    C.assign(E);
    // A*B = E
    // A = E*B^-1
    blas.trsm(f64, .Right, .Upper, 3, 3, 1.0, B.get_ptr(), C.get_ptr(), .{});
    try testing.expectEqual(A.data, C.data);
}