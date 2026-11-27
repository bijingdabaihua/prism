import { SCENE_CARDS } from './catalog';

/**
 * The gallery — the first thing you see. One card per scene; click through to
 * render in real time.
 */
export class GalleryView {
    private root: HTMLElement | null = null;

    public mount(container: HTMLElement): void {
        const root = document.createElement('div');
        root.className = 'gallery';

        const head = document.createElement('header');
        head.className = 'gallery-head';
        head.innerHTML = `
            <h1>Prism</h1>
            <p>WebGPU path tracer · BVH, NEE + MIS and progressive accumulation built from scratch</p>
        `;
        root.appendChild(head);

        // Warn up front without WebGPU, so visitors don't discover it only after clicking through.
        if (!navigator.gpu) {
            const notice = document.createElement('div');
            notice.className = 'gallery-notice';
            notice.textContent =
                'This browser doesn’t support WebGPU, so a scene would open to a static screenshot — switch to Chrome 113+ or Safari 18+ for real-time rendering.';
            root.appendChild(notice);
        }

        const keys = Object.keys(SCENE_CARDS);
        const grid = document.createElement('div');
        grid.className = 'grid';
        for (let i = 0; i < keys.length; i++) {
            grid.appendChild(this.buildCard(keys[i], i));
        }
        root.appendChild(grid);

        const foot = document.createElement('footer');
        foot.className = 'gallery-foot';
        foot.textContent = 'Every image renders in real time: drag with the left button to orbit · scroll to zoom · drag with the right button to pan';
        root.appendChild(foot);

        container.appendChild(root);
        this.root = root;
    }

    public unmount(): void {
        this.root?.remove();
        this.root = null;
    }

    private buildCard(key: string, index: number): HTMLElement {
        const info = SCENE_CARDS[key];
        const card = document.createElement('a');
        card.className = 'card';
        card.href = `#/render/${key}`;
        card.style.setProperty('--tint', String(200 + index * 70));

        const cover = document.createElement('div');
        cover.className = 'card-cover';
        if (info.cover) {
            const img = document.createElement('img');
            img.src = info.cover;
            img.alt = info.title;
            img.loading = 'lazy';
            // Keep the placeholder pattern when the cover is missing instead of the
            // browser's broken-image icon
            img.addEventListener('error', () => img.remove());
            cover.appendChild(img);
        }
        const badge = document.createElement('span');
        badge.className = 'card-badge';
        badge.textContent = 'Real-time →';
        cover.appendChild(badge);
        card.appendChild(cover);

        const body = document.createElement('div');
        body.className = 'card-body';

        const title = document.createElement('h3');
        title.textContent = info.title;
        body.appendChild(title);

        const subtitle = document.createElement('p');
        subtitle.textContent = info.subtitle;
        body.appendChild(subtitle);

        const tags = document.createElement('ul');
        tags.className = 'tags';
        for (const t of info.tags) {
            const li = document.createElement('li');
            li.textContent = t;
            tags.appendChild(li);
        }
        body.appendChild(tags);

        card.appendChild(body);
        return card;
    }
}
