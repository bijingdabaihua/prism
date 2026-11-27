/**
 * Main-thread model handle: contract + implementation.
 *
 * File decoding, displacement and BVH all live in the Worker; this side only passes asset bytes.
 * The factory is `createModel(rpc)`; the implementation class is not exported.
 */

import { Instance } from '../worker/model/instance';
import type { WireInstance } from '../worker/model/instance';

/**
 * Public contract
 *
 * The types and interfaces below are the external API. Check caller impact before changing.
 */

/** Placement for a group of instances; same semantics as `placeInstances` in the scene builder. */
export interface InstancePlacement {
    position: { x: number; y: number; z: number };
    /** Rotation around the X axis in radians, applied before rotationY. */
    rotationX?: number;
    /** Rotation around the Y axis in radians. */
    rotationY?: number;
    scale?: number;
}

/**
 * Model handle — the remote entry point for scene geometry.
 *
 * Both entry points append to the current scene rather than replacing it:
 * - `loadGltf` passes asset-package bytes; parsing, placement and decoding all happen in the
 *   Worker;
 * - `loadInstances` passes a group of IR Instances built by the app on the main thread (e.g.
 *   procedural ground); the handle encodes them into the wire form, and the Worker decodes,
 *   revives and compiles them.
 */
export interface ModelHandle {
    /**
     * Appends every Instance in a glTF asset package.
     *
     * `source` is the asset-package bytes: a self-contained `.glb`, a `.gltf` with embedded
     * resources, or a `.zip` with `.bin` files / textures. URLs are also fetched into a Blob by
     * the caller — the main thread only passes bytes; parsing, unpacking and texture decoding
     * all happen in the Worker.
     *
     * `placement` adds another layer of placement to the whole instance group (same semantics
     * as `placeInstances`). Doing it in the Worker means the main thread does not have to parse
     * an entire model just to place it.
     */
    loadGltf(source: Blob, placement?: InstancePlacement): Promise<void>;
    /**
     * Appends a group of externally built Instances.
     *
     * Typical use is procedural geometry (e.g. ground laid by `MeshBuilder`): geometry and
     * materials are built on the main thread, `Instance.encode` turns them into raw data across
     * the thread boundary, and `Instance.decode` revives them into IR class instances on the
     * Worker side. Texture fields must be decoded `Image` objects (the IR keeps no file paths).
     *
     * One request per batch: nine bricks of ground are nine Instances but compile only once.
     */
    loadInstances(instances: readonly Instance[]): Promise<void>;
    /** Sets the radiance scale for all emissive geometry in the scene; defaults to 1. */
    setEmissionScale(scale: number): Promise<void>;
}

/**
 * Implementation
 *
 * The following is not part of the public contract: message shapes, RPC dependencies and the
 * handle implementation.
 */

/** Worker messages related to models. */
export type ModelRequest =
    | { type: 'model/loadGltf'; id: number; source: Blob; placement?: InstancePlacement }
    | { type: 'model/loadInstances'; id: number; instances: WireInstance[] }
    | { type: 'model/setEmissionScale'; id: number; scale: number };

/** Minimal RPC surface the model handle needs: one request with a return value. */
export interface ModelRpc {
    request<T>(build: (id: number) => ModelRequest): Promise<T>;
}

/** Main-thread model handle: turns both loading entry points into requests with ids. */
class RpcModel implements ModelHandle {
    constructor(private readonly rpc: ModelRpc) {}

    public loadGltf(source: Blob, placement?: InstancePlacement): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'model/loadGltf', id, source, placement }));
    }

    public setEmissionScale(scale: number): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'model/setEmissionScale', id, scale }));
    }

    public loadInstances(instances: readonly Instance[]): Promise<void> {
        return this.rpc.request<void>((id) => ({
            type: 'model/loadInstances',
            id,
            instances: instances.map((instance) => Instance.encode(instance)),
        }));
    }
}

/** Factory for the main-thread model handle; the implementation class is not exported. */
export function createModel(rpc: ModelRpc): ModelHandle {
    return new RpcModel(rpc);
}
