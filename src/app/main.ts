import { SCENE_CARDS } from './catalog';
import { GalleryView } from './gallery';
import { RenderView } from './render-view';
import { UnsupportedView } from './unsupported';

/**
 * Routing: `#/` is the gallery, `#/render/<key>` is the render page.
 *
 * Hash-only — static hosting needs no server rewrite rules.
 */

interface View {
    mount(container: HTMLElement): void | Promise<void>;
    unmount(): void;
}

interface Route {
    name: 'gallery' | 'render';
    key: string;
}

const root = document.getElementById('app')!;

let current: View | null = null;
let currentId = '';

function parseRoute(hash: string): Route {
    const raw = hash.replace(/^#\/?/, '');
    if (!raw) {
        return { name: 'gallery', key: '' };
    }
    const match = raw.match(/^render\/(.+)$/);
    const key = match ? match[1] : raw;
    if (key in SCENE_CARDS) {
        return { name: 'render', key };
    }
    return { name: 'gallery', key: '' };
}

function routeId(route: Route): string {
    return route.name === 'gallery' ? 'gallery' : `render:${route.key}`;
}

function show(hash: string): void {
    const route = parseRoute(hash);
    const id = routeId(route);
    if (id === currentId) return;

    current?.unmount();
    currentId = id;

    if (route.name === 'render') {
        // The gallery is static images and works without WebGPU; only the render page
        // must fall back.
        const card = SCENE_CARDS[route.key];
        if (!navigator.gpu) {
            const view = new UnsupportedView(card);
            current = view;
            void view.mount(root);
            return;
        }

        let view: RenderView;
        view = new RenderView(route.key, () => {
            // API present but no device available (driver blocklist / hardware
            // acceleration off): fall back as well
            if (current !== view) return;
            view.unmount();
            const unsupported = new UnsupportedView(card);
            current = unsupported;
            void unsupported.mount(root);
        });
        current = view;
        void view.mount(root);
    } else {
        const view = new GalleryView();
        current = view;
        view.mount(root);
    }
}

show(location.hash);
window.addEventListener('hashchange', () => show(location.hash));
