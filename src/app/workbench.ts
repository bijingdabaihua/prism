import type { Prism } from '../core/api/prism';
import type { SceneCard } from './catalog';

/** 在当前渲染画布上挂载默认折叠的工作台。 */
export let createWorkbench: (prism: Prism, card: SceneCard, canvas: HTMLCanvasElement, key: string) => HTMLElement;

// ═══════════════════════ 实现层（不导出） ═══════════════════════

createWorkbench = (prism, card, canvas, key) => {
    const workbench = document.createElement('details');
    workbench.className = 'workbench';
    workbench.innerHTML = `
        <summary>调参数</summary>
        <div class="workbench-panel">
            <div class="workbench-actions">
                <button type="button" data-reset>恢复默认</button>
                <button type="button" data-save>保存 PNG</button>
            </div>
            <p role="status"></p>
        </div>
    `;

    const panel = workbench.querySelector('.workbench-panel')!;
    const status = workbench.querySelector('p')!;
    const reset = workbench.querySelector<HTMLButtonElement>('[data-reset]')!;
    const save = workbench.querySelector<HTMLButtonElement>('[data-save]')!;
    let resetEmission: (() => void) | undefined;
    if (card.adjustEmission) {
        const label = document.createElement('label');
        label.innerHTML = `<span>发光亮度 <output>1.00×</output></span>
            <input type="range" min="0" max="3" step="0.05" value="1">`;
        const slider = label.querySelector('input')!;
        const value = label.querySelector('output')!;
        const apply = (): void => {
            const scale = slider.valueAsNumber;
            value.textContent = `${scale.toFixed(2)}×`;
            void prism.model.setEmissionScale(scale).catch((err: unknown) => {
                if (workbench.isConnected) status.textContent = err instanceof Error ? err.message : String(err);
            });
        };
        slider.addEventListener('input', apply);
        resetEmission = () => {
            slider.value = '1';
            apply();
        };
        panel.prepend(label);
    }

    reset.addEventListener('click', () => {
        status.textContent = '';
        prism.camera.set(card.setup.camera);
        resetEmission?.();
    });
    save.addEventListener('click', () => {
        status.textContent = '';
        canvas.toBlob((blob) => {
            if (!workbench.isConnected) return;
            if (!blob) {
                status.textContent = '无法保存当前画面';
                return;
            }
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `prism-${key}.png`;
            link.click();
            window.setTimeout(() => URL.revokeObjectURL(url), 1000);
        }, 'image/png');
    });
    return workbench;
};
