import type { Prism } from '../core/api/prism';
import { requestEmissionCommand } from '../agent/emission-command';

/** 当前发光场景的文字指令面板。 */
export let createWorkbench: (prism: Prism, signal: AbortSignal) => HTMLElement;

// ═══════════════════════ 实现层（不导出） ═══════════════════════

createWorkbench = (prism, signal) => {
    const panel = document.createElement('details');
    panel.className = 'workbench';
    panel.innerHTML = `
        <summary>AI 调场景</summary>
        <form class="workbench-panel">
            <label>场景指令
                <input type="text" maxlength="500" required placeholder="例如：把面板调暗一点">
            </label>
            <button type="submit">执行</button>
            <p role="status"></p>
        </form>
    `;
    const form = panel.querySelector('form')!;
    const input = panel.querySelector('input')!;
    const button = panel.querySelector('button')!;
    const status = panel.querySelector('p')!;
    let current = 1;
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const prompt = input.value.trim();
        if (!prompt || button.disabled) return;
        button.disabled = true;
        status.textContent = '正在理解指令…';
        try {
            const next = await requestEmissionCommand(prompt, current, signal);
            if (signal.aborted) return;
            if (next === null) {
                status.textContent = '当前场景可以调整发光面板的亮度，请试试「把灯调暗一点」。';
                return;
            }
            await prism.model.setEmissionScale(next);
            if (signal.aborted) return;
            current = next;
            status.textContent = `发光亮度已设为 ${Math.round(next * 100)}%。`;
        } catch (err) {
            if (!signal.aborted) status.textContent = err instanceof Error ? err.message : String(err);
        } finally {
            button.disabled = false;
        }
    });
    return panel;
};
