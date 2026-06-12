# tinrs viewer

Zig + Vulkan terrain viewer. A compute shader evaluates the per-tile INR terrain
model on the GPU and feeds a geometry-clipmap renderer, with a synthetic-vision
HUD (heading ribbon, attitude, AGL, TAWS hazard overlay) drawn on top. Flight is
driven by a simple arcade flight model so the camera can fly over the terrain;
the aircraft state is decoupled so real GPS/IMU sensors can swap in later.

## Build and run

Requirements:

- **Zig 0.16**
- A **Vulkan 1.2+** driver
- **SDL3 development libraries** (only for building from source with `zig build
  run`; the prebuilt `zig build dist` tarball links SDL3 statically and needs no
  SDL install)

```bash
cd viewer
zig build run                      # windowed, default tile (n47w122)
```

A bare `zig build run` loads the bundled `n47w122` model from `../assets/planes/`.
To fly with no model at all (procedural terrain, zero assets needed):

```bash
zig build run -- --procedural
```

Other common invocations:

```bash
zig build run -- --tile n47w122 --taws     # with the TAWS terrain-hazard overlay
zig build run -- --headless --profile      # headless benchmark (no window)
zig build run -- --help                    # full flag list
zig build test                             # unit tests
```

## Controls

The menu (Esc, or gamepad Start) captures input while open; the keys below act
while the menu is closed. Camera mode (`M`) selects how you move: **cockpit** and
**chase** fly the aircraft via the flight model; **free** is a 6-DOF fly camera.

### Flight (cockpit / chase)

| Action | Keyboard | Gamepad |
| --- | --- | --- |
| Pitch | `W` / `S` (or Up / Down) | Left stick Y (inverted by default) |
| Roll | `A` / `D` | Left stick X |
| Yaw | `Q` / `E` | Right / Left trigger |
| Throttle | `LShift` / `LCtrl` | R1 / L1 |
| Auto-level toggle | `L` | |

### Free camera (free mode)

| Action | Keyboard | Gamepad |
| --- | --- | --- |
| Move forward / back / strafe | `W` `A` `S` `D` | Left stick |
| Move up / down | `Space` / `LShift` | R1 / L1 |
| Look | Arrow keys | Right stick |
| Fly speed | `-` / `=` | Right / Left trigger |
| Field of view | `[` / `]` | |

### Camera and free-look

| Action | Control |
| --- | --- |
| Cycle camera mode (free -> cockpit -> chase) | `M` or gamepad North (Y) |
| Reset orientation | `R` |
| Free-look glance (cockpit mode) | Hold right mouse, drag right stick, or one-finger drag |
| Recenter glance | Release the look input, gamepad R3, or lose window focus |

### General

| Action | Control |
| --- | --- |
| Toggle HUD | `H` |
| Toggle visual effects (fog / lighting / MSAA) | `V` |
| Open current location in Google Maps | `O` |
| Print diagnostics to the console | `P` |
| Open / close menu | `Esc` or gamepad Start |
| Toggle fullscreen | `F11` or `Alt+Enter` |

### Debug overlays

| Key | Action |
| --- | --- |
| `F1` | Show / hide the dev block (FPS, VRAM, tile residency); hidden by default |
| `F2` | Cycle render mode (normal / wireframe / wireframe+shaded) |
| `F3` | Cycle color overlay (by LOD level / by chunk / by cull state) |
| `F4` | Freeze the camera for inspection (`Shift+F4` keeps tile streaming live) |

### Menu navigation

Arrow keys / `Tab` move focus, `Enter` or `Space` activates, `Esc` backs out,
`[` / `]` switch category tabs. On a gamepad: d-pad navigates, A activates, B
backs out, shoulders switch tabs.

## Command-line flags

Run `--help` for the complete list. The commonly useful ones:

### Terrain and start state

| Flag | Description |
| --- | --- |
| `-t, --tile <name>` | Tile to load, e.g. `n47w122` (default: `n47w122`) |
| `-m, --model <dir>` | Weights directory (default: `../assets/planes`) |
| `-p, --procedural` | Procedural terrain; skip model loading (no assets needed) |
| `--pos <lat,lon,alt_m>` | Start position (default: tile center) |
| `--heading <deg>` | Start heading, 0=N 90=E 180=S 270=W (default: 90) |
| `--pitch <deg>` | Start pitch, + up / - down (default: 0) |
| `--airspeed <arcsec/s>` | Start airspeed (default: 5.0, ~555 km/h; 0 = stalled) |
| `--sensor` | Freeze the aircraft pose (sensor swap-in); free-look stays live |

### Display

| Flag | Description |
| --- | --- |
| `--width <px>` / `--height <px>` | Window size (default: 1280 x 720) |
| `-f, --fullscreen` / `-M, --monitor <n>` | Fullscreen on monitor `n` |
| `-g, --gpu <n>` | Select GPU by candidate index (default: highest-scored) |
| `--fov <deg>` | Field of view (default: 45) |
| `--units <metric\|imperial>` | HUD units (default: metric) |
| `--msaa <1\|2\|4\|8>` | MSAA samples (default: 4; clamped to GPU support) |
| `-u, --no-vsync` | Uncap the frame rate |

### Rendering and overlays

| Flag | Description |
| --- | --- |
| `--taws` | TAWS terrain-hazard color overlay (red/yellow by clearance) |
| `--ring-size <odd>` | Clipmap ring grid dimension per LOD (default: autotuned) |
| `--num-levels <1-12>` | Clipmap LOD level count (default: 6) |
| `--no-effects` | Simple lighting (no fog / ambient / slope shading) |
| `--no-sky` / `--no-fog` / `--no-hdr` | Disable sky / fog / HDR auto-detection |
| `--hud` / `--no-hud` | Force the HUD on / start with it hidden |

### Benchmarking and diagnostics

| Flag | Description |
| --- | --- |
| `-H, --headless` | Run with no window (pure Vulkan; cleanest measurement) |
| `--profile` | Self-calibrating multi-phase benchmark (static + moving) |
| `--benchmark-fly` | Fly the camera during a benchmark (exercises INR compute) |
| `--autotune` | Auto-detect the best ring size for this GPU and save it |
| `-d, --debug` | Verbose logging, GPU candidate enumeration, config dump |
| `-V, --validate` | Vulkan validation layers (core) |

## Settings and persistence

The pause menu (Esc / Start) has **Display**, **Graphics**, and **Input** tabs.
Changes persist to `settings.zon` on interactive runs; CLI flags override the
saved file. The clipmap `ring-size` / `num-levels` are autotuned per GPU and
saved in a small per-GPU config file, so render distance is remembered without
retyping flags. Run with `--autotune` once to calibrate a new GPU.
