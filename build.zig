const std = @import("std");

/// CBLAS system library name, replace with your favorite BLAS implementation
const CBLAS_LIB: []const u8 = "openblas";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const test_blas_mod = b.createModule(.{
        .root_source_file = b.path("src/test_blas.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_blas_mod.linkSystemLibrary(CBLAS_LIB, .{});

    const blas_unit_tests = b.addTest(.{
        .root_module = test_blas_mod,
    });

    const run_blas_unit_tests = b.addRunArtifact(blas_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_blas_unit_tests.step);
}
