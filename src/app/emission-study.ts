import { Instance } from '../core/worker/model/instance';
import { PBRMaterial } from '../core/worker/model/material/material';
import { MeshBuilder } from '../core/worker/model/mesh';

/** An emissive panel lights the ground while a block occludes part of the light. */
export let buildEmissionStudy: () => Instance[];

// ═════════ Implementation layer (not exported) ══════════

buildEmissionStudy = () => {
    const builder = new MeshBuilder();
    type Point = [number, number, number];
    const quad = (a: Point, b: Point, c: Point, d: Point, material: number): void => {
        const point = ([x, y, z]: Point) => ({ x, y, z });
        builder.addQuad([point(a), point(b), point(c), point(d)], material);
    };
    quad([-8, 0, 8], [8, 0, 8], [8, 0, -8], [-8, 0, -8], 0);

    // Tilted panel, low edge facing the block. Both faces emit.
    quad([-1.65, 1.65, -0.55], [-1.65, 1.65, 0.55], [-0.95, 1.05, 0.55], [-0.95, 1.05, -0.55], 1);

    const a: Point = [-0.34, 0, 0.34];
    const b: Point = [0.34, 0, 0.34];
    const c: Point = [0.34, 0, -0.34];
    const d: Point = [-0.34, 0, -0.34];
    const top = ([x, , z]: Point): Point => [x, 0.68, z];
    quad(top(a), top(b), top(c), top(d), 2);
    for (const [p, q] of [[a, b], [b, c], [c, d], [d, a]]) {
        quad(q, p, top(p), top(q), 2);
    }

    const floor = new PBRMaterial({ baseColor: { r: 0.54, g: 0.56, b: 0.61, a: 1 }, roughness: 0.92 });
    const panel = new PBRMaterial({ baseColor: { r: 0, g: 0, b: 0, a: 1 }, emission: { r: 12, g: 8, b: 4, a: 1 } });
    const block = new PBRMaterial({ baseColor: { r: 0.38, g: 0.24, b: 0.15, a: 1 }, roughness: 0.88 });
    return [new Instance(builder.build([floor, panel, block], 'emission-study'))];
};
