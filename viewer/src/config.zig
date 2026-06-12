//! Umbrella for the config subsystem. Re-exports each file in `config/`
//! as a namespace so callers can do `@import("config.zig").options.Config`.

pub const config_file = @import("config/config_file.zig");
pub const options = @import("config/options.zig");
pub const settings = @import("config/settings.zig");
pub const sysinfo = @import("config/sysinfo.zig");

test {
    _ = config_file;
    _ = options;
    _ = settings;
    _ = sysinfo;
}
