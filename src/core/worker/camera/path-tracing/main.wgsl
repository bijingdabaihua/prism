// Path tracing with NEE (Next Event Estimation) — explicit direct light sampling + indirect scattering

/** Bounce after which Russian roulette starts (0-based). */
const RR_START_BOUNCE: u32 = 4u;

// Camera ray generation — pixel coordinates + sub-pixel jitter for AA.
// Consumes 2 LDS dimensions (pixel_x, pixel_y jitter), plus 2 more when the
// lens has an aperture (thin-lens depth of field).
//
// Thin lens: a pixel determines a point **on the focal plane**, not a direction.
// The ray starts at a random point on the aperture disk and aims at that point —
// the larger the disk, the more objects outside the focal plane blur. When the
// aperture radius is 0 the disk degenerates to a point, the expression reduces to
// a pinhole, and no extra sample is taken.
fn generate_ray(pixel_x: u32, pixel_y: u32, seed: u32, dim: ptr<function, u32>) -> Ray {
  let eye = camera.camera_position.xyz;
  let jx = lds(seed, u.frame_index, *dim); *dim += 1u;
  let jy = lds(seed, u.frame_index, *dim); *dim += 1u;
  let jittered_x = f32(pixel_x) + jx;
  let jittered_y = f32(pixel_y) + jy;
  let ndc_x = jittered_x / f32(u.resolution.x) * 2.0 - 1.0;
  let ndc_y = jittered_y / f32(u.resolution.y) * 2.0 - 1.0;
  let clip = vec4(ndc_x, -ndc_y, 1.0, 1.0);
  let world = camera.inv_view_projection * clip;
  let direction = normalize(world.xyz / world.w - eye);

  if (camera.aperture_radius <= 0.0 || camera.focus_distance <= 0.0) {
    return Ray(eye, direction);
  }

  // Uniform point on the aperture disk: distribute the radius by sqrt(r),
  // otherwise points bunch at the center.
  let r1 = lds(seed, u.frame_index, *dim); *dim += 1u;
  let r2 = lds(seed, u.frame_index, *dim); *dim += 1u;
  let radius = sqrt(r1) * camera.aperture_radius;
  let angle = 2.0 * 3.14159265 * r2;

  // Camera right/up basis. When the view direction is straight down, cross
  // degenerates; fall back to the world X axis.
  let right_raw = cross(direction, vec3(0.0, 1.0, 0.0));
  let right = select(vec3(1.0, 0.0, 0.0), normalize(right_raw), length(right_raw) > 1e-6);
  let up = cross(right, direction);

  let offset = right * (radius * cos(angle)) + up * (radius * sin(angle));
  let focal_point = eye + direction * camera.focus_distance;
  let origin = eye + offset;

  return Ray(origin, normalize(focal_point - origin));
}

// Scatter — BSDF-scatter result at a vertex.
//
//   contrib : radiance from this vertex (environment on escape), already
//             weighted; caller adds  L += beta * scat.contrib.
//   weight  : BSDF weight multiplier for beta on the next bounce.
//   alive   : false stops the path (miss).
//   next    : surface at the scattered direction; first-iteration hole
//             is filled by the caller, subsequent ones from scatter's probe.
struct Scatter {
  contrib: vec3<f32>,
  weight: vec3<f32>,
  alive: bool,
  next: SurfaceInfo,
}

// scatter — BSDF estimator at a vertex.
//
// Determines what happens when a ray lands on this surface.  The light transport
// leaving a vertex is estimated by:
//
//   1. NEE (direct_environment) — pick an environment direction, trace a shadow
//      ray, eval BSDF.  MIS weight: pdf_nee² / (pdf_nee² + pdf_bsdf²).
//
//   2. BSDF scatter (this function) — sample a direction from the BSDF, trace a
//      probe ray.  If the probe escapes into the environment, that radiance is
//      the BSDF estimator.  MIS weight: pdf_bsdf² / Σ pdf².
//
// Both estimators contribute to the same transport path — without MIS, energy
// doubles.  The power heuristic reconciles them.
//
// Cases — miss is resolved by path_trace's zero-bounce term (before scatter is
// reached), so `hit` here is always a surface:
//   probe escapes to env → env radiance (BSDF-weighted), terminate.
//   probe hits surface   → alive, continue with `next`.
// Smooth dielectric. Sampling by Fresnel cancels the selected lobe's probability.
fn glass_direction(incident: vec3<f32>, normal: vec3<f32>, eta: f32, sample: f32) -> vec3<f32> {
  if (eta == 1.0) { return incident; }
  let cos_i = clamp(dot(-incident, normal), 0.0, 1.0);
  let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
  if (sin_t2 >= 1.0) { return reflect(incident, normal); }
  let cos_t = sqrt(1.0 - sin_t2);
  let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  if (sample < 0.5 * (rs * rs + rp * rp)) { return reflect(incident, normal); }
  return refract(incident, normal, eta);
}

fn scatter_glass(hit: SurfaceInfo, sample: f32) -> Scatter {
  var direction = glass_direction(-hit.wo, hit.smooth_normal, hit.eta, sample);
  // Interpolated normals can send a grazing ray to the wrong side of the triangle.
  let reflected = dot(direction, hit.smooth_normal) > 0.0;
  if (reflected != (dot(direction, hit.geo_normal) > 0.0)) {
    direction = glass_direction(-hit.wo, hit.geo_normal, hit.eta, sample);
  }
  let exits_surface = dot(direction, hit.geo_normal) > 0.0;
  let offset = select(-hit.geo_normal, hit.geo_normal, exits_surface) * 0.0001;
  let weight = vec3(select(hit.eta * hit.eta, 1.0, exits_surface));
  let next_hit = intersect_surface(Ray(hit.position + offset, normalize(direction)));
  if (!next_hit.hit) {
    return Scatter(weight * env_radiance(direction), vec3(0.0), false, next_hit);
  }
  return Scatter(weight * next_hit.emission, weight, true, next_hit);
}

fn scatter(hit: SurfaceInfo, seed: u32, dim: ptr<function, u32>) -> Scatter {
  // hit is always a surface here — miss is handled by path_trace's zero-bounce
  // term, which breaks before calling scatter.
  let r1 = lds(seed, u.frame_index, *dim); *dim += 1u;
  let r2 = lds(seed, u.frame_index, *dim); *dim += 1u;
  let r3 = lds(seed, u.frame_index, *dim); *dim += 1u;
  if (hit.transmission > 0.0) {
    let lobe = lds(seed, u.frame_index, *dim); *dim += 1u;
    if (lobe < hit.transmission) { return scatter_glass(hit, r1); }
  }
  let bsdf = sample_bsdf(hit, r1, r2, r3);
  // Sampled below the horizon (pdf = 0): the sample is undefined, so terminate
  // the path.
  if (bsdf.pdf <= 0.0) { return Scatter(vec3(0.0), vec3(0.0), false, hit); }
  let weight = bsdf.f * abs(dot(hit.normal, bsdf.wo)) / max(bsdf.pdf, 1e-6);

  // Probe toward scattered direction — result reused as `next` by the caller
  let probe_origin = hit.position + hit.geo_normal * 0.0001;
  let next_hit = intersect_surface(Ray(probe_origin, bsdf.wo));

  if (!next_hit.hit) {
    // Probe escaped into the environment — add its radiance here (BSDF-weighted)
    // and terminate.  Deferring this to the next bounce dropped it when the
    // bounce budget ran out (e.g. max_bounces=0 → box exteriors went black).
    // MIS: environment NEE may also sample the same direction; the power heuristic
    // splits the contribution, otherwise energy is counted twice.
    // Specular samples use a roughness-based prefiltered mip (smooth surfaces blur
    // only slightly); cosine samples always use mip 0 — see the BSDFSample.specular comment.
    let env = select(env_radiance(bsdf.wo), env_radiance_rough(bsdf.wo, hit.roughness), bsdf.specular);
    let pdf_env = environment_pdf(bsdf.wo);
    let mis_w = mis_weight(bsdf.pdf, pdf_env);
    // First AO application: single-bounce light from the sky/environment is
    // attenuated by the occlusion factor (that's what AO means: sky visibility).
    // The sun's NEE term isn't here; it remains unchanged in direct_light.
    return Scatter(mis_w * weight * env * hit.ao, vec3(0.0), false, next_hit);
  }
  var emitted = vec3(0.0);
  if (any(next_hit.emission > vec3(0.0))) {
    let pdf_light = emitter_pdf(hit.position, next_hit.position, next_hit.geo_normal);
    emitted = next_hit.emission * weight * mis_weight(bsdf.pdf, pdf_light);
  }
  return Scatter(emitted, weight, true, next_hit);
}


// intersect_surface — cast a ray and evaluate material at the hit point.
fn intersect_surface(ray: Ray) -> SurfaceInfo {
  return compute_surface(ray, intersect_scene(ray));
}

// Path tracing loop — unidirectional path tracing with NEE + MIS.
//
// Each bounce first adds the zero-bounce radiance (environment on escape) —
// this needs no scattering.  Then the two estimators for light transport to a
// surface are summed into L:
//   1. NEE  (direct_environment) — sample the environment, shadow ray, MIS-weighted.
//   2. BSDF (scatter)        — sample BSDF direction; if the probe escapes, its
//                               environment radiance is the BSDF estimator.
//
// scatter also returns the next surface (via probe, saving one BVH traversal
// per bounce) and the throughput weight.
fn path_trace(ray: Ray, seed: u32, dim: ptr<function, u32>) -> vec3<f32> {
  var beta = vec3(1.0);
  var L = vec3(0.0);
  var hit = intersect_surface(ray);
  L += hit.emission;

  for (var bounce: u32 = 0u; bounce <= u.max_bounces; bounce++) {
    // Zero-bounce radiance: a ray escaping into the environment contributes its
    // radiance without scattering.
    if (!hit.hit) {
      L += beta * env_radiance(-hit.wo);
      break;
    }

    // NEE estimator
    if (hit.transmission < 1.0) {
      let direct = direct_environment(hit, seed, dim) + direct_emitter(hit, seed, dim);
      L += beta * (1.0 - hit.transmission) * direct;
    }

    // BSDF estimator
    let scat = scatter(hit, seed, dim);
    L += beta * scat.contrib;
    if (!scat.alive) { break; }
    // AO multiplies only paths that keep bouncing: direct light (NEE and the BSDF
    // hit this bounce) is already in L and unaffected. This term treats macro
    // occlusion as visibility for indirect light.
    beta *= scat.weight * hit.ao;

    hit = scat.next;

    // Russian roulette: after the 4th bounce, decide whether to continue based on
    // throughput; terminated paths are compensated by 1/p. The expected value is
    // unchanged, but it avoids the energy loss of a hard max_bounces cutoff (with
    // albedo 0.73 in a closed box, cutting off after 5 bounces loses about 20% of
    // the energy).
    if (bounce >= RR_START_BOUNCE) {
      let p = clamp(max(beta.r, max(beta.g, beta.b)), 0.05, 1.0);
      let rr = lds(seed, u.frame_index, *dim); *dim += 1u;
      if (rr > p) { break; }
      beta /= p;
    }
  }

  return L;
}

// Progressive accumulation
//
// Partial sampling: each frame writes only a contiguous run of the order table
// (dispatch_count pixels); the remaining pixels are preserved by the CPU-side
// texture copy each frame. So "how many times this pixel was sampled" can no
// longer use the frame index — it is determined by its position in the order
// table. See main.
fn accumulate(coordinate: vec2<i32>, radiance: vec3<f32>, previous_count: u32) {
  let prev = textureLoad(previous_accumulation_texture, coordinate, 0).rgb;
  let count = f32(previous_count);
  let blended = (prev * count + radiance) / (count + 1.0);
  textureStore(out_texture, coordinate, vec4(blended, 1.0));
}

// 1D dispatch: one thread = one pixel in the order table.
//
// Global sample index sample_index = cursor + slot; it crosses a sweep boundary
// at most once (the CPU guarantees dispatch_count ≤ pixel_count). The order table
// is a random permutation with no repeats over a full sweep, so every time
// pixel_count samples are crossed, all pixels have been sampled exactly once
// more; therefore previous_count = completed sweeps + (whether this one crossed
// into the next sweep), with no per-pixel counter needed.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x;
  if (slot >= u.dispatch_count) { return; }

  let sample_index = u.cursor + slot;
  let previous_count = u.sweep + select(0u, 1u, sample_index >= u.pixel_count);
  let pixel = pixel_order[sample_index % u.pixel_count];

  let pixel_x = pixel % u.resolution.x;
  let pixel_y = pixel / u.resolution.x;
  let coordinate = vec2<i32>(i32(pixel_x), i32(pixel_y));

  let seed = pixel_x ^ (pixel_y << 16u);
  var dim: u32 = 0u;
  let ray = generate_ray(pixel_x, pixel_y, seed, &dim);
  let L = path_trace(ray, seed, &dim);
  accumulate(coordinate, L, previous_count);
}
