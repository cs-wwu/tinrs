//! Resolve the platform-appropriate config directory for tinrs:
//!   Linux:   $XDG_CONFIG_HOME/tinrs  or  $HOME/.config/tinrs
//!   macOS:   $HOME/Library/Application Support/tinrs
//!   Windows: %APPDATA%\tinrs
//!
//! This is where `settings.zon` lives (see config/settings.zig). The old
//! per-GPU `viewer.zon` autotune store was folded into `settings.zon`
//! (render-distance is a single `rendering.ring_size`/`num_levels` there now),
//! so this file is just the directory resolver.

const std = @import("std");
const builtin = @import("builtin");

/// Returns the platform-correct config dir for tinrs (e.g.
/// "/home/x/.config/tinrs"). Caller frees. Returns null when no env var
/// suitable for the current platform is set.
pub fn configDirAlloc(
    allocator: std.mem.Allocator,
    env: *const std.process.Environ.Map,
) std.mem.Allocator.Error!?[]u8 {
    return switch (builtin.os.tag) {
        .windows => if (env.get("APPDATA") orelse env.get("LOCALAPPDATA")) |base|
            try std.fs.path.join(allocator, &.{ base, "tinrs" })
        else
            null,
        .macos => if (env.get("HOME")) |h|
            try std.fs.path.join(allocator, &.{ h, "Library", "Application Support", "tinrs" })
        else
            null,
        else => if (env.get("XDG_CONFIG_HOME")) |x|
            try std.fs.path.join(allocator, &.{ x, "tinrs" })
        else if (env.get("HOME")) |h|
            try std.fs.path.join(allocator, &.{ h, ".config", "tinrs" })
        else
            null,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "configDirAlloc returns null when relevant env vars are missing" {
    var env: std.process.Environ.Map = .{
        .array_hash_map = .{},
        .allocator = testing.allocator,
    };
    defer env.deinit();
    const result = try configDirAlloc(testing.allocator, &env);
    try testing.expect(result == null);
}

test "configDirAlloc builds a path when env vars are present" {
    var env: std.process.Environ.Map = .{
        .array_hash_map = .{},
        .allocator = testing.allocator,
    };
    defer env.deinit();

    switch (builtin.os.tag) {
        .windows => try env.put("APPDATA", "C:\\Users\\x\\AppData\\Roaming"),
        .macos => try env.put("HOME", "/Users/x"),
        else => try env.put("HOME", "/home/x"),
    }

    const result = try configDirAlloc(testing.allocator, &env);
    try testing.expect(result != null);
    defer testing.allocator.free(result.?);
    try testing.expect(std.mem.endsWith(u8, result.?, "tinrs"));
}
