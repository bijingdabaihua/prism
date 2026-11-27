// Shared structs and bindings for path tracing

struct Uniforms {
  resolution: vec2<u32>,
  frame_index: u32,      // Frame index: feeds only LDS, not the running average
  max_bounces: u32,
  // ── Partial sampling ──
  dispatch_count: u32,   // pixels dispatched this frame (≤ pixel_count)
  cursor: u32,           // pixels sampled so far in the current sweep [0, pixel_count)
  sweep: u32,            // completed full sweeps (= lower bound on each pixel's sample count)
  pixel_count: u32,      // width * height
  emission_scale: f32,
}

struct CameraUniforms {
  inv_view_projection: mat4x4<f32>,
  camera_position: vec4<f32>,
  resolution: vec4<f32>,
  fov: f32,
  // Entrance pupil radius (meters). 0 = pinhole.
  aperture_radius: f32,
  // Focus distance (meters, measured along the view direction). 0 = infinity.
  focus_distance: f32,
}

struct Ray { origin: vec3<f32>, direction: vec3<f32> }
struct BVHResult { t: f32, u: f32, v: f32, triangle_index: u32, instance_index: u32 }

struct Instance {
  blas_node_offset: f32,
  transform: mat4x3<f32>,
  inv_transform: mat4x3<f32>,
}

const epsilon = 1e-6;

@group(0) @binding(0) var<uniform> u: Uniforms;
// TLAS + BLAS merged node table: [TLAS ...][BLAS ...], 8 floats per node.
// BLAS node offsets already include the TLAS length at packing time (see gpu/upload.ts).
@group(0) @binding(2) var<storage, read> nodes: array<f32>;
// Pixel sampling order: a random permutation of 0..pixel_count-1 (see gpu/pixel-order.ts).
@group(0) @binding(3) var<storage, read> pixel_order: array<u32>;
@group(0) @binding(4) var<storage, read> geometry: array<f32>;
@group(0) @binding(5) var<storage, read> triangles: array<u32>;
@group(0) @binding(6) var<storage, read> materials: array<f32>;
@group(0) @binding(20) var<storage, read> emitters: array<f32>;
@group(0) @binding(7) var<storage, read> instances: array<Instance>;
@group(0) @binding(8) var out_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var environment_texture: texture_2d<f32>;
// Material textures: three texture_2d_array bindings; material slots store
// layer indices (see TextureSet).
@group(0) @binding(10) var albedo_texture: texture_2d_array<f32>;
@group(0) @binding(11) var normal_texture: texture_2d_array<f32>;
@group(0) @binding(12) var roughness_texture: texture_2d_array<f32>;
// AO map (R channel): attenuates indirect light only; direct light hitting the
// surface is unaffected. See beta in main.wgsl.
@group(0) @binding(19) var occlusion_texture: texture_2d_array<f32>;
@group(0) @binding(17) var texture_sampler: sampler;
// Environment-map sampler: u wraps, v clamps. The material texture sampler is
// repeat/repeat — the environment's v is the polar angle, so wrapping goes
// straight from the north pole to the south.
@group(0) @binding(21) var environment_sampler: sampler;
@group(0) @binding(13) var previous_accumulation_texture: texture_2d<f32>;
@group(0) @binding(15) var<uniform> camera: CameraUniforms;
// Environment luminance CDF (layout in engine/background.ts)
@group(0) @binding(16) var<storage, read> env_distribution: array<f32>;
