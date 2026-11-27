import type { SceneCard } from './catalog';

/**
 * Fallback page: stands in for the render page when the browser lacks WebGPU.
 *
 * A landing page must answer three questions and nothing else matters: what was
 * here, why it cannot be seen now, and how to see it. So it shows an image
 * rendered from the same source, a one-line reason, and a support matrix.
 */
const BROWSERS: ReadonlyArray<readonly [string, string]> = [
    ['Chrome / Edge', '113 and later, enabled by default'],
    ['Safari', '18 and later'],
    ['Firefox', 'Enable dom.webgpu.enabled in about:config'],
];

export class UnsupportedView {
    private root: HTMLElement | null = null;
    private readonly card: SceneCard;

    constructor(card: SceneCard) {
        this.card = card;
    }

    public mount(container: HTMLElement): void {
        const root = document.createElement('div');
        root.className = 'unsupported';

        const back = document.createElement('a');
        back.className = 'unsupported-back';
        back.href = '#/';
        back.textContent = '← All scenes';
        root.appendChild(back);

        const head = document.createElement('header');
        head.className = 'unsupported-head';
        const title = document.createElement('h1');
        title.textContent = 'This browser cannot run WebGPU yet';
        head.appendChild(title);
        const lede = document.createElement('p');
        lede.textContent =
            'Prism computes every frame live on the GPU, and this site has no pre-rendered version. The image below shows what the same code produces in a supported browser.';
        head.appendChild(lede);
        root.appendChild(head);

        const figure = document.createElement('figure');
        figure.className = 'unsupported-figure';
        if (this.card.cover) {
            const img = document.createElement('img');
            img.src = this.card.cover;
            img.alt = this.card.title;
            figure.appendChild(img);
        }
        const caption = document.createElement('figcaption');
        caption.textContent = `${this.card.title} — ${this.card.subtitle}`;
        figure.appendChild(caption);
        root.appendChild(figure);

        const table = document.createElement('table');
        table.className = 'unsupported-table';
        for (const [name, how] of BROWSERS) {
            const row = document.createElement('tr');
            const th = document.createElement('th');
            th.textContent = name;
            const td = document.createElement('td');
            td.textContent = how;
            row.appendChild(th);
            row.appendChild(td);
            table.appendChild(row);
        }
        root.appendChild(table);

        const note = document.createElement('p');
        note.className = 'unsupported-note';
        note.textContent =
            'If you are already on a recent Chrome, open chrome://gpu and search for WebGPU to confirm its status is Enabled — a browser version alone is not enough when an enterprise policy or experimental flag has turned it off.';
        root.appendChild(note);

        container.appendChild(root);
        this.root = root;
    }

    public unmount(): void {
        this.root?.remove();
        this.root = null;
    }
}
