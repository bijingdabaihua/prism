@group(0) @binding(0) var input_texture: texture_2d<f32>;

// The only place in the whole pipeline that does display mapping: linear HDR →
// tone mapping → sRGB encoding. No scaling is inserted in between: the environment
// map already provides scene radiance, ACES squeezes it into the display, and an
// extra "exposure" multiplier would only turn a physical quantity back into an
// arbitrary knob.

// Lens look: all scenes share the same "lens"; its parameters don't follow the
// scene. Vignette and grain are applied after tone mapping (in display space),
// matching a real camera's imaging order.
const VIGNETTE_STRENGTH: f32 = 0.45;
const GRAIN_STRENGTH: f32 = 0.012;

// ACES fit (Narkowicz 2015): compresses unbounded HDR values into [0,1] with a
// smooth highlight rolloff.
fn aces_tone_map(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let v = clamp(c, vec3(0.0), vec3(1.0));
  let low = v * 12.92;
  let high = 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055;
  return select(high, low, v <= vec3(0.0031308));
}

// Static grain: hashed from pixel coordinates, constant across frames — otherwise
// the whole image would flicker, and a converged still would drown in grain. Its
// strength only breaks up banding in the shadows without changing tonal relationships.
fn grain_hash(p: vec2<f32>) -> f32 {
  var p3 = fract(p.xyx * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  // Render size matches the canvas one-to-one, so just take whole pixels; anything
  // out of range (transition frames where the canvas is larger than the texture)
  // returns 0 under textureLoad's out-of-bounds rules, so no wrong content is drawn.
  let coord = vec2<i32>(position.xy);
  let hdr = textureLoad(input_texture, coord, 0).rgb;

  var color = aces_tone_map(hdr);

  // Vignette: darker farther from the image center, down to about 78% at the corners
  let uv = position.xy / vec2<f32>(textureDimensions(input_texture));
  let offset = uv - vec2<f32>(0.5);
  color *= 1.0 - VIGNETTE_STRENGTH * dot(offset, offset);

  let srgb = linear_to_srgb(color);
  let grain = (grain_hash(position.xy) - 0.5) * GRAIN_STRENGTH;
  return vec4(clamp(srgb + grain, vec3(0.0), vec3(1.0)), 1.0);
}
