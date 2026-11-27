// Shading: PBR material with environment reflection + material texture sampling

// LDS — stateless low-discrepancy sampler via PCG-hashed (seed, sample, dim).
//
// Each (sample, dim) pair produces a deterministic pseudo-random value; the
// seed varies per pixel.  Multiplication by large primes preserves all bits
// unlike << which truncates; the Wang finalizer provides high-quality
// avalanche mixing.

fn lds(seed: u32, sample: u32, dim: u32) -> f32 {
  var v = seed;
  v = v ^ (sample * 0x9E3779B9u);   // golden-ratio prime
  v = v ^ (dim * 0x517CC1B7u);      // another large prime
  v = v * 0x85EBCA6Bu;  // Wang hash finalizer: 3 rounds of xor-shift-mul
  v = v ^ (v >> 16u);
  v = v * 0xC2B2AE35u;
  v = v ^ (v >> 13u);
  v = v * 0x27D4EB2Du;
  v = v ^ (v >> 16u);
  return f32(v >> 8u) / f32(0xFFFFFFu);
}

fn lds2(seed: u32, sample: u32, dim: u32) -> vec2<f32> {
  return vec2(lds(seed, sample, dim), lds(seed, sample, dim + 1u));
}

// SurfaceInfo — material data at a hit point, used by bounce loop

struct SurfaceInfo {
  position: vec3<f32>,
  normal: vec3<f32>,
  // The triangle's own face normal, flipped toward the ray. Shading uses the
  // interpolated normal above; shadow-ray offsets use this one — on scanned
  // models the vertex normals differ a lot from the true surface at high
  // curvature, and offsetting with them pushes rays into adjacent triangles,
  // showing up as fixed black dots on the surface.
  geo_normal: vec3<f32>,
  wo: vec3<f32>,           // direction from surface toward incoming ray
  albedo: vec3<f32>,
  roughness: f32,
  metallic: f32,
  // Ambient occlusion (0..1): attenuates only indirect light that keeps bouncing
  // from this point; direct light isn't multiplied.
  ao: f32,
  hit: bool,
  transmission: f32,
  eta: f32,
  smooth_normal: vec3<f32>,
  emission: vec3<f32>,
}

// Evaluate material at a hit point — no lighting, used by bounce loop

// Resolve the GPU material slot for a triangle: absolute material index baked
// into triangles[*][3] at merge time.
fn material_slot(global_triangle_index: u32) -> u32 {
  return triangles[global_triangle_index * 4u + 3u];
}

fn compute_surface(ray: Ray, hit: BVHResult) -> SurfaceInfo {
  // Miss — no geometry at all; store ray direction in wo for env map sampling
  if (hit.t <= 0.0) {
    return SurfaceInfo(vec3(0.0), vec3(0.0), vec3(0.0), -ray.direction, vec3(0.0), 0.0, 0.0, 1.0, false, 0.0, 1.0, vec3(0.0), vec3(0.0));
  }

  let inst = instances[hit.instance_index];
  let o = material_slot(u32(hit.triangle_index)) * 33u;

  var base_color = vec3(materials[o+0u], materials[o+1u], materials[o+2u]);
  // metallic / roughness are glTF factors: with a texture they act as multipliers;
  // without one they are the whole value.
  let metallic_factor = materials[o+3u];
  let roughness_factor = materials[o+4u];
  var metallic = metallic_factor;
  var roughness = roughness_factor;
  let normal_scale = materials[o+5u];

  // Read triangle vertices
  let triangle_data_offset = hit.triangle_index * 4u;
  let i0 = triangles[triangle_data_offset];
  let i1 = triangles[triangle_data_offset+1u];
  let i2 = triangles[triangle_data_offset+2u];

  let v0 = read_vertex(i0);
  let v1 = read_vertex(i1);
  let v2 = read_vertex(i2);

  let w = 1.0 - hit.u - hit.v;
  let uv = w * v0.uv + hit.u * v1.uv + hit.v * v2.uv;
  // Defensive: a NaN/zero barycentric sum would make normalize() NaN (normalize(0) = NaN).
  // Fall back to the geometric face normal so the hit normal is always valid.
  var local_normal = w * v0.normal + hit.u * v1.normal + hit.v * v2.normal;
  if (!(dot(local_normal, local_normal) > 0.0)) {
    local_normal = cross(v1.position - v0.position, v2.position - v0.position);
  }
  local_normal = normalize(local_normal);

  let local_tangent = w * v0.tangent + hit.u * v1.tangent + hit.v * v2.tangent;
  let world_tangent = normalize((inst.transform * vec4<f32>(local_tangent.xyz, 0.0)).xyz);
  let world_normal = normalize((inst.transform * vec4<f32>(local_normal, 0.0)).xyz);
  let facing_normal = select(-world_normal, world_normal, dot(world_normal, -ray.direction) > 0.0);

  var geo_local = cross(v1.position - v0.position, v2.position - v0.position);
  if (!(dot(geo_local, geo_local) > 0.0)) { geo_local = local_normal; }
  let world_geo = normalize((inst.transform * vec4<f32>(geo_local, 0.0)).xyz);
  let entering = dot(world_geo, -ray.direction) > 0.0;
  let facing_geo = select(-world_geo, world_geo, entering);

  let tangent_axis = normalize(world_tangent - facing_normal * dot(facing_normal, world_tangent));
  let bitangent_axis = cross(facing_normal, tangent_axis) * local_tangent.w;

  // Texture sampling: each material slot is [layer, scale, has, _] (see compile/packing.ts).
  //
  // LOD must be estimated: path tracing has no screen-space derivatives, so
  // textureSample can't be used. The key is that a pixel footprint isn't a circle:
  // when the view grazes the ground, one pixel covers several meters along the
  // view direction but only a few millimeters perpendicular to it. The old code
  // computed one isotropic size from "angle to the normal", dropping mips on both
  // axes together, so distant ground blurred into horizontal streaks — that's
  // where "not enough resolution" on grazing surfaces came from. So compute per
  // axis along u/v: the minor axis sets the mip, the major axis is recovered with
  // multiple taps (see sample_texture).
  //
  // The world footprint along axis a is step / sqrt(1 - (d·a)²): the more grazing
  // the view along that axis, the farther a screen pixel spans along it. Clamped
  // to 20x so it doesn't blow up near the horizon.
  let pixel_angle = clamp(
    2.0 * tan(camera.fov * 0.5) / max(camera.resolution.y, 1.0),
    1e-6,
    0.5,
  );
  let step = hit.t * pixel_angle;
  let d_tan = dot(tangent_axis, ray.direction);
  let d_bit = dot(bitangent_axis, ray.direction);
  let foot_t = step / sqrt(max(1.0 - d_tan * d_tan, 0.0025));
  let foot_b = step / sqrt(max(1.0 - d_bit * d_bit, 0.0025));

  var normal_perturbation = vec3<f32>(0.0, 0.0, 1.0);
  if (materials[o + 10u] > 0.5) {
    let scale = materials[o + 9u];
    base_color = sample_texture(albedo_texture, materials[o + 8u], scale, uv, foot_t, foot_b).rgb;
  }
  // Packed metallic-roughness (glTF convention / Poly Haven arm maps): G = roughness,
  // B = metallic. Only in this version can metallic actually vary with a texture —
  // before it was scalar only, so a steel plate with both painted and bare metal
  // areas came out wrong.
  if (materials[o + 26u] > 0.5) {
    let scale = materials[o + 25u];
    let mr = sample_texture(roughness_texture, materials[o + 24u], scale, uv, foot_t, foot_b);
    roughness = mr.g * roughness_factor;
    metallic = mr.b * metallic_factor;
  }
  // Single-channel roughness map: when present alongside the packed map, it
  // overrides roughness only.
  if (materials[o + 14u] > 0.5) {
    let scale = materials[o + 13u];
    roughness = sample_texture(roughness_texture, materials[o + 12u], scale, uv, foot_t, foot_b).r * roughness_factor;
  }
  if (materials[o + 18u] > 0.5) {
    let scale = materials[o + 17u];
    let ns = sample_texture(normal_texture, materials[o + 16u], scale, uv, foot_t, foot_b).rgb;
    normal_perturbation = normalize(ns * 2.0 - 1.0);
    normal_perturbation = vec3(normal_perturbation.xy * normal_scale, normal_perturbation.z);
  }

  // AO: material slots follow the same [layer, scale, has, _] convention as the
  // other three maps. glTF occlusion only has an R channel; the importer copies R
  // into RGB, so taking .r here is enough.
  var ao = 1.0;
  if (materials[o + 22u] > 0.5) {
    let occ = sample_texture(occlusion_texture, materials[o + 20u], materials[o + 21u], uv, foot_t, foot_b).r;
    ao = mix(1.0, occ, clamp(materials[o + 6u], 0.0, 1.0));
  }

  let mapped_normal = normalize(
    tangent_axis * normal_perturbation.x +
    bitangent_axis * normal_perturbation.y +
    facing_normal * normal_perturbation.z
  );

  let hit_pos = ray.origin + ray.direction * hit.t;

  return SurfaceInfo(
    hit_pos,
    mapped_normal,
    facing_geo,
    -ray.direction,
    base_color,
    roughness,
    metallic,
    ao,
    true,
    materials[o + 28u] * (1.0 - clamp(metallic, 0.0, 1.0)),
    select(materials[o + 29u], 1.0 / materials[o + 29u], entering),
    facing_normal,
    u.emission_scale * vec3(materials[o + 30u], materials[o + 31u], materials[o + 32u]),
  );
}

// TBN — build orthonormal basis from a unit normal

struct TBN { u: vec3<f32>, v: vec3<f32>, w: vec3<f32> }

fn build_tbn(n: vec3<f32>) -> TBN {
  let w = normalize(n);
  // Pick a reference axis guaranteed non-parallel to w
  let a = normalize(select(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), abs(w.x) > 0.9));
  let u = normalize(cross(w, a));
  let v = cross(w, u);
  return TBN(u, v, w);
}

// sample_hemisphere_cosine — cosine-weighted direction in hemisphere around n

fn sample_hemisphere_cosine(n: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
  // Validate n: if n is NaN or zero, build_tbn will break — fall back to +Y
  if (n.x != n.x || n.y != n.y || n.z != n.z || dot(n, n) < 1e-12) { return vec3(0.0, 1.0, 0.0); }
  let c1 = clamp(r1, 0.0, 1.0);
  let c2 = clamp(r2, 0.0, 1.0);
  let tbn = build_tbn(n);
  let phi = c1 * 6.2831853;
  let cos_theta = sqrt(max(1.0 - c2, 0.0));
  let sin_theta = sqrt(max(c2, 0.0));
  let dir = tbn.u * sin_theta * cos(phi) + tbn.v * sin_theta * sin(phi) + tbn.w * cos_theta;
  return normalize(dir);
}

// BSDF sampling: mixture of a diffuse lobe (Lambert) + a specular lobe (GGX)
//
// Sampling picks one lobe according to specular_probability; the returned pdf must
// be the **mixture** density p·pdf_spec + (1-p)·pdf_diff — that is the true
// density of this sampling process at wo, and NEE's MIS uses the same number.
//
// Sampling and evaluation share evaluate_bsdf, so their pdfs can't diverge. The
// old code sampled pure cosine but evaluated a GGX mixture; the two disagreed, so
// the MIS weights didn't sum to 1.

struct BSDFSample {
  wo: vec3<f32>,
  f: vec3<f32>,
  pdf: f32,
  /// Whether this direction was sampled from the specular or cosine lobe. When
  /// escaping to the environment, only specular samples merit a roughness-based
  /// prefiltered mip — cosine samples want a cosine-weighted hemisphere average,
  /// and multiplying them by a mip blurred into a global average doubles the
  /// ground brightness.
  specular: bool,
}

/// Specular-lobe sampling probability. sample_bsdf and evaluate_bsdf must use the
/// same value. Dielectrics (metallic = 0) reserve 20% for specular: their true
/// contribution is only a few percent, and giving it all to specular starves
/// cosine sampling, noticeably worsening diffuse noise in indoor scenes.
fn specular_probability(metallic: f32) -> f32 {
  return clamp(mix(0.2, 1.0, metallic), 0.05, 1.0);
}

/// Smith geometry term G1 (one-directional masking). The VNDF pdf needs it.
fn smith_g1(n_dot_v: f32, alpha2: f32) -> f32 {
  let c = max(n_dot_v, 1e-4);
  return 2.0 * c / (c + sqrt(alpha2 * (1.0 - c * c) + c * c));
}

/// VNDF sampling — Heitz 2018, "Sampling the GGX Distribution of Visible Normals".
///
/// Compared with NDF sampling by D(h)·(n·h), it samples only the part of the
/// microsurface visible from the view: at grazing angles it doesn't waste samples
/// on half-vectors below the horizon, so noise is noticeably lower.
///
/// Tangent space in, tangent space out (normal = +Z); ve is the view direction.
fn sample_ggx_vndf(ve: vec3<f32>, alpha: f32, r1: f32, r2: f32) -> vec3<f32> {
  let vh = normalize(vec3(alpha * ve.x, alpha * ve.y, max(ve.z, 1e-4)));

  let lensq = vh.x * vh.x + vh.y * vh.y;
  var t1 = vec3(1.0, 0.0, 0.0);
  if (lensq > 1e-12) { t1 = vec3(-vh.y, vh.x, 0.0) * inverseSqrt(lensq); }
  let t2 = cross(vh, t1);

  let r = sqrt(clamp(r1, 0.0, 1.0));
  let phi = 6.2831853 * r2;
  let p1 = r * cos(phi);
  var p2 = r * sin(phi);
  // Stretch the lower half of the disk to avoid microsurfaces facing away from the view
  let s = 0.5 * (1.0 + vh.z);
  p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * p2;

  let nh = p1 * t1 + p2 * t2 + sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * vh;
  return normalize(vec3(alpha * nh.x, alpha * nh.y, max(nh.z, 1e-6)));
}

/// GGX half-vector sampling in world space. alpha = roughness².
fn sample_ggx_half_vector(n: vec3<f32>, wo: vec3<f32>, alpha: f32, r1: f32, r2: f32) -> vec3<f32> {
  let tbn = build_tbn(n);
  let ve = vec3(dot(wo, tbn.u), dot(wo, tbn.v), dot(wo, tbn.w));
  let nh = sample_ggx_vndf(ve, alpha, r1, r2);
  return normalize(tbn.u * nh.x + tbn.v * nh.y + tbn.w * nh.z);
}

fn sample_bsdf(surf: SurfaceInfo, r1: f32, r2: f32, r3: f32) -> BSDFSample {
  let n = surf.normal;
  let alpha = max(surf.roughness * surf.roughness, 0.001);
  let spec_prob = specular_probability(surf.metallic);

  var wo: vec3<f32>;
  let specular = r3 < spec_prob;
  if (specular) {
    let h = sample_ggx_half_vector(n, surf.wo, alpha, r1, r2);
    wo = normalize(reflect(-surf.wo, h));
    // Reflected below the horizon: the density there is 0, so the sample must be
    // discarded; otherwise f·cos/pdf divides out a huge weight and becomes a firefly.
    if (dot(wo, n) <= 0.0) { return BSDFSample(vec3(0.0, 1.0, 0.0), vec3(0.0), 0.0, true); }
  } else {
    wo = sample_hemisphere_cosine(n, r1, r2);
  }

  let val = evaluate_bsdf(-wo, surf.wo, n, surf.metallic, surf.roughness, surf.albedo);
  return BSDFSample(wo, val.f, val.pdf, specular);
}

// Evaluvation — given wi (toward surface), wo (away from surface),
// compute f(wi, wo) and pdf(wi→wo) for explicit light sampling / MIS.

struct BSDFResult {
  f: vec3<f32>,
  pdf: f32,
}

fn evaluate_bsdf(wi: vec3<f32>, wo: vec3<f32>, n: vec3<f32>, metallic: f32, roughness: f32, albedo: vec3<f32>) -> BSDFResult {
  let wi_std = -wi;
  let n_dot_wi = max(dot(n, wi_std), 0.001);
  let n_dot_wo = max(dot(n, wo), 0.001);

  // Half-vector: undefined when wi and wo are nearly opposite (grazing), where the
  // BRDF is meaningless.
  let hsum = wi_std + wo;
  if (dot(hsum, hsum) < 1e-12) { return BSDFResult(vec3(0.0), 0.0); }
  let h = normalize(hsum);
  let h_dot_n = dot(n, h);
  let n_dot_h = max(h_dot_n, 0.001);
  let h_dot_wo = max(dot(h, wo), 0.001);
  // Half-vector below the surface: the VNDF sampler never produces such a
  // half-vector, so both the specular lobe's f and pdf must be zero; otherwise NEE
  // overestimates the BSDF pdf at grazing angles.
  let spec_visible = select(0.0, 1.0, h_dot_n > 0.0);

  let a = max(roughness * roughness, 0.001);
  let a2 = a * a;

  let f0 = mix(vec3(0.04), albedo, metallic);
  // Schlick's angle is between the view direction and the half-vector, not the normal
  let fresnel = f0 + (vec3(1.0) - f0) * pow(1.0 - h_dot_wo, 5.0);

  let d = a2 / (3.14159265 * pow(n_dot_h * n_dot_h * (a2 - 1.0) + 1.0, 2.0));

  let ggxv = n_dot_wo * sqrt((n_dot_wi * n_dot_wi) * (1.0 - a2) + a2);
  let ggxl = n_dot_wi * sqrt((n_dot_wo * n_dot_wo) * (1.0 - a2) + a2);
  let g = 2.0 * n_dot_wi * n_dot_wo / max(ggxv + ggxl, 1e-8);

  let f_spec = spec_visible * fresnel * d * g / (4.0 * n_dot_wi * n_dot_wo + 1e-8);
  let f_diff = albedo * (1.0 - metallic) / 3.14159265;

  // Solid-angle pdf of VNDF sampling (Heitz 2018): p(wo) = D(h)·G1(wo) / (4·(n·wo)).
  // Must use the same sampling process as sample_ggx_half_vector, or weights and MIS go wrong.
  let pdf_spec = spec_visible * d * smith_g1(n_dot_wo, a2) / (4.0 * n_dot_wo);
  let pdf_diff = n_dot_wi / 3.14159265;

  let spec_prob = specular_probability(metallic);

  return BSDFResult(
    f_spec + f_diff,
    pdf_spec * spec_prob + pdf_diff * (1.0 - spec_prob),
  );
}

// Normal offset for shadow rays. It must exceed the intersection epsilon (1e-6) to
// avoid self-intersection, yet stay well below the smallest face-to-face gap in the
// scene — 0.1 mm.

const SHADOW_EPSILON = 1e-4;

// MIS power heuristic — beta=2: the estimator with higher PDF dominates.
// pdf_a = PDF of the estimator being weighted, pdf_b = competing PDF.

fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
  let a2 = pdf_a * pdf_a;
  return a2 / max(a2 + pdf_b * pdf_b, 1e-12);
}

// NEE: environment-light sampling
//
// The environment is an emitter at infinity — no point, only directions: sample a
// direction from the luminance CDF, trace a shadow ray, and if visible take that
// direction's radiance directly.
//
// Without this, the sun could only be hit "by chance" through diffuse cosine
// sampling, and the occasional hit produces fireflies amplified by 1/pdf. With NEE
// the sun direction is sampled directly every frame, and the rare BSDF-side hits
// are suppressed by MIS, so fireflies disappear.

fn direct_environment(surf: SurfaceInfo, seed: u32, dim: ptr<function, u32>) -> vec3<f32> {
  let r1 = lds(seed, u.frame_index, *dim); *dim += 1u;
  let r2 = lds(seed, u.frame_index, *dim); *dim += 1u;
  let sample = sample_environment_light(r1, r2);

  // Sample only directions in the upper hemisphere (normal side)
  let cos_surf = dot(surf.normal, sample.direction);
  if (cos_surf <= 0.0) { return vec3(0.0); }

  // The environment is at infinity: hitting any surface counts as occluded
  let shadow_origin = surf.position + surf.geo_normal * SHADOW_EPSILON;
  let shadow_hit = intersect_scene(Ray(shadow_origin, sample.direction));
  if (shadow_hit.t > 0.0) { return vec3(0.0); }

  let bsdf_val = evaluate_bsdf(
    -sample.direction, surf.wo, surf.normal, surf.metallic, surf.roughness, surf.albedo
  );
  if (bsdf_val.pdf <= 0.0) { return vec3(0.0); }

  // Both pdfs are in solid-angle measure, so they can be compared directly.
  let mis_w = mis_weight(sample.pdf, bsdf_val.pdf);
  let radiance = env_texel(sample.direction);

  return mis_w * bsdf_val.f * radiance * cos_surf / max(sample.pdf, 1e-12);
}

// Converts area sampling into a solid-angle PDF, using the same measure as BSDF sampling.
fn emitter_pdf(origin: vec3<f32>, point: vec3<f32>, normal: vec3<f32>) -> f32 {
  let delta = point - origin;
  let distance_squared = dot(delta, delta);
  let cosine = abs(dot(normalize(delta), normal));
  return distance_squared / max(emitters[1] * cosine, 1e-12);
}

fn direct_emitter(surf: SurfaceInfo, seed: u32, dim: ptr<function, u32>) -> vec3<f32> {
  let count = u32(emitters[0]);
  if (count == 0u || u.emission_scale == 0.0) { return vec3(0.0); }
  let area = lds(seed, u.frame_index, *dim) * emitters[1]; *dim += 1u;
  let r1 = lds(seed, u.frame_index, *dim); *dim += 1u;
  let r2 = lds(seed, u.frame_index, *dim); *dim += 1u;
  var lo = 0u;
  var hi = count - 1u;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if (area < emitters[2u + mid * 13u + 9u]) { hi = mid; }
    else { lo = mid + 1u; }
  }
  let i = 2u + lo * 13u;
  let a = vec3(emitters[i], emitters[i + 1u], emitters[i + 2u]);
  let b = vec3(emitters[i + 3u], emitters[i + 4u], emitters[i + 5u]);
  let c = vec3(emitters[i + 6u], emitters[i + 7u], emitters[i + 8u]);
  let r = sqrt(r1);
  let point = (1.0 - r) * a + r * (1.0 - r2) * b + r * r2 * c;
  let delta = point - surf.position;
  if (dot(delta, delta) < 1e-10) { return vec3(0.0); }
  let wi = normalize(delta);
  let cosine = dot(surf.normal, wi);
  if (cosine <= 0.0 || dot(surf.geo_normal, wi) <= 0.0) { return vec3(0.0); }

  let origin = surf.position + surf.geo_normal * SHADOW_EPSILON;
  let to_light = point - origin;
  let blocker = intersect_scene(Ray(origin, normalize(to_light)));
  if (blocker.t > 0.0 && blocker.t < length(to_light) - SHADOW_EPSILON) { return vec3(0.0); }

  let pdf = emitter_pdf(surf.position, point, normalize(cross(b - a, c - a)));
  let bsdf = evaluate_bsdf(-wi, surf.wo, surf.normal, surf.metallic, surf.roughness, surf.albedo);
  let emission = u.emission_scale * vec3(emitters[i + 10u], emitters[i + 11u], emitters[i + 12u]);
  return bsdf.f * emission * cosine * mis_weight(pdf, bsdf.pdf) / max(pdf, 1e-12);
}

// Environment / atlas samplers (used by the bounce loop)

fn environment_uv(direction: vec3<f32>) -> vec2<f32> {
  let phi = atan2(direction.z, direction.x);
  let theta = acos(clamp(direction.y, -1.0, 1.0));
  return vec2(phi / 6.2831853 + 0.5, 1.0 - theta / 3.14159265);
}

/// Camera ray escaping straight into the environment: mip 0 with bilinear filtering.
fn env_radiance(direction: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(environment_texture, environment_sampler, environment_uv(direction), 0.0).rgb;
}

/// For NEE: fetch exact texels, using the same function as luminance-CDF sampling.
fn env_texel(direction: vec3<f32>) -> vec3<f32> {
  let uv = environment_uv(direction);
  let size = textureDimensions(environment_texture, 0);
  // Clamp to a valid texel: when uv lands exactly at 1.0 the coordinate goes out of bounds.
  let coordinate = clamp(
    vec2<i32>(uv * vec2<f32>(size)),
    vec2<i32>(0),
    vec2<i32>(size) - vec2<i32>(1),
  );
  return textureLoad(environment_texture, coordinate, 0).rgb;
}

/// For BSDF escape directions: fetch a prefiltered mip based on roughness.
///
/// The GGX lobe covers a texel width on the environment map of about
/// 2α / (2π / width) = α·width/π (α = roughness²); log2 of that is the mip level.
/// Without this, each sample on a smooth surface fetches a single texel — a glazed
/// reflection is then a 2k point cloud, adjacent samples jump between bright and
/// dark, and convergence is very slow.
///
/// Using a different mip than NEE doesn't affect unbiasedness: MIS weights look
/// only at the two pdfs, not at brightness; and what's estimated here is the
/// in-lobe environment average, for which a blurred mip is exactly the approximation.
fn env_radiance_rough(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
  let width = f32(textureDimensions(environment_texture, 0).x);
  let alpha = max(roughness * roughness, 1e-4);
  let lod = clamp(log2(max(alpha * width / 3.14159265, 1.0)), 0.0, 12.0);
  return textureSampleLevel(environment_texture, environment_sampler, environment_uv(direction), lod).rgb;
}

// Environment importance sampling
//
// Environment-map per-pixel weight = luminance × sinθ (normalized within a row);
// the marginal CDF is then normalized by row weights. Sampling: pick a row from
// the marginal CDF, pick a column from that row's conditional CDF, then take the
// direction at the pixel center.
//
// Solid-angle conversion: for equirectangular (u,v) the Jacobian to solid angle is
// 2π²·sinθ (u covers 2π, v covers π), so pdf_ω = p_row · p_col · w·h / (2π² · sinθ).
// Sampling and pdf use the same formula, keeping MIS weights consistent.

struct EnvLightSample {
  direction: vec3<f32>,
  pdf: f32,
}

fn sample_environment_light(r1: f32, r2: f32) -> EnvLightSample {
  let size = textureDimensions(environment_texture, 0);
  let w = size.x;
  let h = size.y;
  let stride = w + 1u;
  let marginal_offset = h * stride;

  // Pick a row (marginal CDF index y means "cumulative over the first y rows")
  var row_lo: u32 = 0u;
  var row_hi = h - 1u;
  loop {
    if (row_lo >= row_hi) { break; }
    let mid = (row_lo + row_hi) / 2u;
    if (env_distribution[marginal_offset + mid + 1u] <= r1) { row_lo = mid + 1u; }
    else { row_hi = mid; }
  }
  let row = row_lo;
  let row_base = row * stride;

  // Pick a column (row-local conditional CDF)
  var col_lo: u32 = 0u;
  var col_hi = w - 1u;
  loop {
    if (col_lo >= col_hi) { break; }
    let mid = (col_lo + col_hi) / 2u;
    if (env_distribution[row_base + mid + 1u] <= r2) { col_lo = mid + 1u; }
    else { col_hi = mid; }
  }
  let col = col_lo;

  let row_prob = env_distribution[marginal_offset + row + 1u] - env_distribution[marginal_offset + row];
  let col_prob = env_distribution[row_base + col + 1u] - env_distribution[row_base + col];

  // Pixel center → direction (inverse of environment_uv)
  let theta = 3.14159265 * (1.0 - (f32(row) + 0.5) / f32(h));
  let phi = 6.2831853 * ((f32(col) + 0.5) / f32(w) - 0.5);
  let sin_theta = sin(theta);
  let direction = vec3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));

  let wh = f32(w) * f32(h);
  let pdf = row_prob * col_prob * wh / (19.7392088 * max(sin_theta, 1e-6));

  return EnvLightSample(direction, pdf);
}

/**
 * PDF of an arbitrary direction under the environment distribution
 * (solid-angle measure), used by the BSDF estimator for MIS.
 */
fn environment_pdf(direction: vec3<f32>) -> f32 {
  let size = textureDimensions(environment_texture, 0);
  let w = size.x;
  let h = size.y;
  let stride = w + 1u;

  let uv = environment_uv(direction);
  let col = min(u32(uv.x * f32(w)), w - 1u);
  let row = min(u32(uv.y * f32(h)), h - 1u);
  let marginal_offset = h * stride;

  let row_prob = env_distribution[marginal_offset + row + 1u] - env_distribution[marginal_offset + row];
  let col_prob = env_distribution[row * stride + col + 1u] - env_distribution[row * stride + col];

  let theta = 3.14159265 * (1.0 - (f32(row) + 0.5) / f32(h));
  let wh = f32(w) * f32(h);
  return row_prob * col_prob * wh / (19.7392088 * max(sin(theta), 1e-6));
}

/**
 * Fetches one layer from the material texture array with an explicit LOD and does
 * anisotropic sampling along the axis with the longer footprint.
 *
 * foot_t / foot_b are the pixel footprint lengths in meters along the surface's
 * two tangent directions. Converted to texels, the minor axis sets the mip (so
 * detail isn't over-blurred), and the ratio of major to minor determines how many
 * taps to spread along the major axis. A wall viewed head-on has ratio 1 and
 * degenerates to a single sample; only grazing ground takes multiple taps.
 */
fn sample_texture(
  texture: texture_2d_array<f32>,
  layer: f32,
  scale: f32,
  uv: vec2<f32>,
  foot_t: f32,
  foot_b: f32,
) -> vec4<f32> {
  let size = f32(textureDimensions(texture, 0).x);
  let texels_t = max(foot_t * scale * size, 1.0);
  let texels_b = max(foot_b * scale * size, 1.0);
  let major = max(texels_t, texels_b);
  let minor = min(texels_t, texels_b);
  // Cap at 4 taps: clamp even at larger ratios, otherwise per-sample cost is unbounded.
  let taps = i32(clamp(ceil(major / minor), 1.0, 4.0));
  let dir = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), texels_t >= texels_b);

  var sum = vec4<f32>(0.0);
  for (var i = 0; i < taps; i++) {
    // Taps spread evenly over the footprint: with taps = 1 the offset is exactly 0
    let offset = (f32(i) + 0.5) / f32(taps) - 0.5;
    sum += textureSampleLevel(
      texture,
      texture_sampler,
      uv * scale + dir * offset * major / size,
      i32(layer),
      log2(minor),
    );
  }
  return sum / f32(taps);
}
