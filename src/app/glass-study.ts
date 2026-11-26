import { Instance } from '../core/worker/model/instance';
import { PBRMaterial } from '../core/worker/model/material/material';
import { Mesh, MeshBuilder } from '../core/worker/model/mesh';

/** 棋盘和封闭球体，用于观察光滑玻璃对格线的折射。 */
export let buildGlassStudy: () => Instance[];

// ═══════════════════════ 实现层（不导出） ═══════════════════════

buildGlassStudy = () => {
    const board = new MeshBuilder();
    const tile = 0.38;
    for (let x = -14; x < 14; x++) {
        for (let z = -14; z < 14; z++) {
            const x0 = x * tile;
            const x1 = (x + 1) * tile;
            const z0 = z * tile;
            const z1 = (z + 1) * tile;
            board.addQuad([
                { x: x0, y: 0, z: z1 },
                { x: x1, y: 0, z: z1 },
                { x: x1, y: 0, z: z0 },
                { x: x0, y: 0, z: z0 },
            ], (x + z) & 1);
        }
    }

    const tiles = [0.06, 0.74].map((value) => new PBRMaterial({
        baseColor: { r: value, g: value, b: value, a: 1 },
        roughness: 0.85,
    }));
    return [new Instance(board.build(tiles, 'checkerboard')), new Instance(buildSphere())];
};

function buildSphere(): Mesh {
    const segments = 96;
    const rings = 48;
    const radius = 0.6;
    const vertices: number[] = [];
    const triangles: number[] = [];
    vertices.push(0, 2 * radius, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1);
    for (let ring = 1; ring < rings; ring++) {
        const theta = Math.PI * ring / rings;
        const y = Math.cos(theta);
        for (let segment = 0; segment < segments; segment++) {
            const phi = 2 * Math.PI * segment / segments;
            const x = Math.sin(theta) * Math.cos(phi);
            const z = Math.sin(theta) * Math.sin(phi);
            vertices.push(
                radius * x, radius * (y + 1), radius * z,
                x, y, z, segment / segments, ring / rings,
                -Math.sin(phi), 0, Math.cos(phi), 1,
            );
        }
    }
    const south = vertices.length / 12;
    vertices.push(0, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, 1);

    for (let segment = 0; segment < segments; segment++) {
        const next = (segment + 1) % segments;
        triangles.push(0, 1 + next, 1 + segment, 0);
        for (let ring = 0; ring < rings - 2; ring++) {
            const a = 1 + ring * segments + segment;
            const b = 1 + ring * segments + next;
            triangles.push(a, b, a + segments, 0, b, b + segments, a + segments, 0);
        }
        const last = 1 + (rings - 2) * segments;
        triangles.push(last + segment, last + next, south, 0);
    }

    const glass = new PBRMaterial({ transmission: 1, ior: 1.5, roughness: 0 });
    return new Mesh(new Float32Array(vertices), new Uint32Array(triangles), [glass], 'glass-sphere');
}
