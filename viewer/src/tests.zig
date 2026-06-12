//! Aggregate test root for the file-import tree.
//!
//! `zig build test` compiles this as a SINGLE test binary so every test runs
//! exactly once. Zig deduplicates tests within one compilation, so files shared
//! across subsystems (e.g. terrain/coords.zig pulled by both terrain and app,
//! app/camera.zig pulled by both app and config) are compiled and run once here
//! instead of once per per-subsystem test binary. This is Ghostty's pattern: one
//! file-tree test root, aggregated by `_ = @import` umbrellas.
//!
//! Referencing each subsystem umbrella transitively pulls in every file-tree
//! test, because each umbrella's own `test {}` block `_ = @import`s its files.
//! New files are picked up by adding them to their subdir umbrella; a new
//! top-level subsystem gets one `_ = @import` line below.
//!
//! NOT here:
//!   - The `math` and `ui` MODULES: reached elsewhere via @import("math") /
//!     @import("ui") module imports, their tests run only when the module is the
//!     addTest root, so build.zig gives each its own pure (Vulkan-free) binary.
//!   - main.zig: the exe entry point, compile-validated by `zig build` (which
//!     builds the full exe graph), not by `zig build test`.

test {
    _ = @import("app.zig");
    _ = @import("render.zig");
    _ = @import("terrain.zig");
    _ = @import("bench.zig");
    _ = @import("config.zig");
    _ = @import("scene.zig");
    _ = @import("hud.zig");
    _ = @import("session.zig");
    _ = @import("window.zig");
}
