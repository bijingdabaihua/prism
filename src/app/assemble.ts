import type { ModelHandle, Prism } from '../core/api/prism';
import type { SceneCard, SceneGeometry } from './catalog';
import { buildEmissionStudy } from './emission-study';
import { buildGlassStudy } from './glass-study';
import { buildTiledGround } from './ground';

/**
 * Loads a scene: camera placement → environment map + orientation → geometry.
 *
 * The order is part of the scene's semantics: the camera placement sets the
 * composition first, the environment map decides where the light comes from,
 * and the geometry lands last. Prism exposes three independent handles with no
 * "whole scene" interface; this function just collects the business layer's one
 * canonical order in a single place, and the render page calls it.
 *
 * The catalog is pure data (`catalog.ts`); importers and the network are only
 * touched here — reading cards in the gallery never pulls glTF / blend /
 * decoders into its load path.
 */
export async function loadScene(prism: Prism, card: SceneCard): Promise<void> {
    prism.camera.set(card.setup.camera);

    // The environment map reaches the Worker as a Blob: the main thread only
    // fetches bytes, decoding stays in the Worker. setImage builds it in the
    // source orientation; setRotation turns it when a scene needs another heading.
    if (card.setup.env === null) {
        const blackHdr = new Blob(['#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n', new Uint8Array(8)]);
        await prism.background.setImage(blackHdr);
    } else {
        await prism.background.setImage(await fetchBlob(card.setup.env));
        await prism.background.setRotation(card.setup.envRotation ?? 0);
    }

    await loadGeometry(prism.model, card.geometry);
}

/**
 * Loads geometry in catalog order. Procedural geometry goes through the
 * instance entry point:
 * - `tiled-ground`: the main thread builds a batch of Instances and sends them
 *   to the Worker in one request;
 * - `gltf`: only the asset package bytes are handed over; parsing, texture
 *   decoding and placement all happen in the Worker.
 */
async function loadGeometry(model: ModelHandle, geometry: readonly SceneGeometry[]): Promise<void> {
    for (const object of geometry) {
        if (object.type === 'tiled-ground') {
            const material = await fetchBlob(object.material);
            await model.loadInstances(await buildTiledGround(material, object));
        } else if (object.type === 'emission-study') {
            await model.loadInstances(buildEmissionStudy());
        } else if (object.type === 'glass-study') {
            await model.loadInstances(buildGlassStudy());
        } else {
            await model.loadGltf(await fetchBlob(object.url), object.placement);
        }
    }
}

/** URL → Blob. Parsing in the Worker starts from a Blob; fetching bytes is the caller's job. */
async function fetchBlob(url: string): Promise<Blob> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url} (${resp.status})`);
    return resp.blob();
}
