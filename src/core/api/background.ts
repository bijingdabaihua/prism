/**
 * Main-thread environment-map handle. The main thread only supplies bytes;
 * format detection, decoding and upload all happen in the Worker.
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ══════════════════ Contract layer (exports) ═════════════════

/**
 * Creates the main-thread environment-map handle; returns the contract interface, and the
 * implementation class is not exported.
 * The Blob is handed to the Worker for decoding and upload — the main thread only supplies bytes.
 * This is only a declaration; the assignment lives in the implementation layer.
 */
export let createBackground: (rpc: BackgroundRpc) => BackgroundHandle;

/**
 * Environment-map handle — the remote entry point for the environment map.
 */
export interface BackgroundHandle {
    /**
     * Sets the environment map (Blob). The Worker handles format detection, decoding and upload.
     * Every change starts from the original orientation; call `setRotation` if you need a heading.
     * When the Promise resolves, the texture and CDF are in place.
     */
    setImage(blob: Blob): Promise<void>;
    /**
     * Rotates the current environment map (radians around +y). Requires a prior `setImage`;
     * when done the texture and CDF are rebuilt together, and the Promise resolves once in effect.
     */
    setRotation(rotation: number): Promise<void>;
}

/**
 * The minimal RPC surface the environment-map handle depends on: one request with a return value.
 * `build` produces the request message to send to the Worker (shape defined in this layer's
 * `BackgroundRequest`), opaque to the contract layer.
 */
export interface BackgroundRpc {
    request<T>(build: (id: number) => BackgroundRequest): Promise<T>;
}

/**
 * Worker messages related to environment lighting.
 * The message shape is not part of the public contract, but the Prism message-dispatch union
 * needs to enumerate it, so it is exported and belongs in this layer.
 */
export type BackgroundRequest =
    | { type: 'background/setImage'; id: number; blob: Blob }
    | { type: 'background/setRotation'; id: number; rotation: number };

// ═════════════ Implementation layer (not exported) ════════════

/** Main-thread implementation of the environment-map handle. */
class RpcBackground implements BackgroundHandle {
    constructor(private readonly rpc: BackgroundRpc) {}

    public setImage(blob: Blob): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'background/setImage', id, blob }));
    }

    public setRotation(rotation: number): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'background/setRotation', id, rotation }));
    }
}

/** Assignment for the contract layer's factory entry point. */
createBackground = (rpc) => new RpcBackground(rpc);
