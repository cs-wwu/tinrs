//! Single source of truth for Vulkan and SDL3 C imports.
//! All files import c and vk from here to avoid duplicate @cImport opaque types.

pub const c = @cImport({
    // MinGW's <string.h>/<wchar.h> emit FORTIFY-source inline wrappers around
    // wcscat/wcscpy when _FORTIFY_SOURCE > 0. Zig 0.16's translate-c trips on
    // those inlines (declares an unused `extern_local_wcscat_s` struct), so the
    // x86_64-windows-gnu cross-build fails before it even sees any of our code.
    // Forcing the level to 0 silences the inline emission and is harmless: we
    // never call those CRT helpers ourselves.
    @cDefine("_FORTIFY_SOURCE", "0");
    @cInclude("SDL3/SDL.h");
    // SDL_vulkan.h forward-declares VkSurfaceKHR / VkInstance via its own
    // VK_DEFINE_*_HANDLE macros, so we don't need to pull in <vulkan/vulkan.h>
    // here. Skipping it keeps the cross-compile from depending on a system
    // Vulkan SDK and avoids dragging another large header through translate-c.
    @cInclude("SDL3/SDL_vulkan.h");
});
pub const vk = @import("vulkan");
