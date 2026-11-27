import type { Image } from '../image';
import type { PBRMaterial, TextureKind } from './material';
import type { MaterialTexSlots, TexRef } from './packing';
import { TextureSet } from './texture-set';
import { packPBRMaterial, PBR_MAT_FLOATS } from './packing';

/**
 * GPU material store — holds the material SSBO.
 *
 * One material record = PBR_MAT_FLOATS floats; the shader indexes it by the absolute slot
 * stored in triangles. The same material object is packed once and shares one slot when
 * referenced by several meshes.
 *
 * The material table also owns its TextureSet: a TexRef in the table is an atlas layer
 * number, and the two are halves of one piece of data; holding them apart would just add
 * a constructor argument.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ══════════════════ Contract layer (exports) ═════════════════

/**
 * Class binding for MaterialStore: construction creates an empty TextureSet.
 *
 * register() may be called repeatedly as scenes are appended; compile() rebuilds the SSBO
 * from the currently registered materials and returns the previous SSBO — Model releases it
 * after the GPU is done with it.
 *
 * Only declarations here; assignment is in the implementation layer.
 */
export let MaterialStore: {
    new (device: GPUDevice): MaterialStore;
};

/**
 * GPU material store: the material SSBO plus its texture atlas, with slots deduplicated by
 * material object.
 */
export interface MaterialStore {
    /**
     * Atlas for material sampling; PathTracePass takes views of the four arrays directly
     * from here.
     */
    readonly textures: TextureSet;

    /**
     * Returns the material's absolute slot in the material table; registering the same
     * object again returns the same slot.
     */
    register(material: PBRMaterial): number;
    /**
     * Writes all registered materials into a new SSBO and returns the previous one (null on
     * first compile).
     *
     * The material table only grows, so rebuilding the whole block when a scene is appended
     * is the simplest approach. The old buffer may still be referenced by submitted bind
     * groups; the caller must wait for the GPU before destroying it.
     */
    compile(): GPUBuffer | null;
    buffer(): GPUBuffer;
    destroy(): void;
}

// ═════════════ Implementation layer (not exported) ════════════

class MaterialStoreImpl implements MaterialStore {
    public readonly textures: TextureSet;
    private ssbo!: GPUBuffer;
    private materials: Float32Array[] = [];
    private readonly device: GPUDevice;
    private readonly slots = new Map<PBRMaterial, number>();

    constructor(device: GPUDevice) {
        this.device = device;
        this.textures = new TextureSet(device);
    }

    public register(material: PBRMaterial): number {
        const cached = this.slots.get(material);
        if (cached !== undefined) return cached;
        const index = this.materials.length;
        const tex = this.resolveTextures(material);
        this.materials.push(packPBRMaterial(material, tex));
        this.slots.set(material, index);
        return index;
    }

    public compile(): GPUBuffer | null {
        if (this.materials.length === 0) {
            throw new Error('MaterialStore: the scene has no materials');
        }
        const data = new Float32Array(this.materials.length * PBR_MAT_FLOATS);
        for (let i = 0; i < this.materials.length; i++) {
            data.set(this.materials[i], i * PBR_MAT_FLOATS);
        }
        const previous = this.ssbo ?? null;
        const ssbo = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'scene-materials',
        });
        this.device.queue.writeBuffer(ssbo, 0, data as any);
        this.ssbo = ssbo;
        return previous;
    }

    public buffer(): GPUBuffer {
        return this.ssbo;
    }

    public destroy(): void {
        this.ssbo?.destroy();
        this.textures.destroy();
    }

    /**
     * Material textures → texture_2d_array layer numbers.
     *
     * Textures in the IR are already decoded Images, so no lookup is needed here — decoding
     * and deduplication happen on the importer side (loadTexture in image.ts); the compiler
     * layer only allocates layer numbers.
     */
    private resolveTextures(material: PBRMaterial): MaterialTexSlots {
        return {
            baseColor: this.texRef(material.baseColorTexture, 'albedo'),
            roughness: this.texRef(material.roughnessTexture, 'roughness'),
            normal: this.texRef(material.normalTexture, 'normal'),
            occlusion: this.texRef(material.occlusionTexture, 'occlusion'),
            // The packed map is a data map (G = rough, B = metal) and shares an array with
            // single-channel roughness: both are rgba8unorm with independent channels, so
            // there is no need for another binding.
            metallicRoughness: this.texRef(material.metallicRoughnessTexture, 'roughness'),
        };
    }

    /**
     * Repeats per meter, converted from the texture's own meters per tile — geometry UVs
     * are in meters.
     */
    private texRef(image: Image | undefined, kind: TextureKind): TexRef | undefined {
        if (!image) return undefined;
        return { layer: this.textures.register(kind, image), scale: 1 / (image.meters ?? 1), has: 1 };
    }
}

/** Assignment of the contract-layer class binding. */
MaterialStore = MaterialStoreImpl;
