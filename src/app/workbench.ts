import type { Prism } from '../core/api/prism';
import { requestEmissionCommand } from '../agent/emission-command';

/** Text command panel for the current emissive scene. */
export let createWorkbench: (prism: Prism, signal: AbortSignal) => HTMLElement;

// ═════════ Implementation layer (not exported) ══════════

createWorkbench = (prism, signal) => {
    const panel = document.createElement('details');
    panel.className = 'workbench';
    panel.innerHTML = `
        <summary>AI scene control</summary>
        <form class="workbench-panel">
            <label>Scene command
                <input type="text" maxlength="500" required placeholder="e.g. dim the panel">
            </label>
            <button type="submit">Run</button>
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
        status.textContent = 'Understanding the command…';
        try {
            const next = await requestEmissionCommand(prompt, current, signal);
            if (signal.aborted) return;
            if (next === null) {
                status.textContent = 'This scene can adjust the brightness of the emissive panel. Try "dim the light a little."';
                return;
            }
            await prism.model.setEmissionScale(next);
            if (signal.aborted) return;
            current = next;
            status.textContent = `Emission brightness set to ${Math.round(next * 100)}%.`;
        } catch (err) {
            if (!signal.aborted) status.textContent = err instanceof Error ? err.message : String(err);
        } finally {
            button.disabled = false;
        }
    });
    return panel;
};
