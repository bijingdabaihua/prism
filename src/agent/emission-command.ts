/// <reference types="vite/client" />

/** The command the Agent may execute for the emission scene. null means it cannot be executed. */
export function readEmissionCommand(value: unknown): number | null {
    if (typeof value !== 'object' || value === null || !('operation' in value) || !('scale' in value)) return null;
    const { operation, scale } = value;
    if (operation !== 'setEmissionScale' || typeof scale !== 'number' || !Number.isFinite(scale)) return null;
    return scale >= 0 && scale <= 3 ? scale : null;
}

export async function requestEmissionCommand(prompt: string, current: number, signal: AbortSignal): Promise<number | null> {
    const key = import.meta.env.VITE_PUBLIC_DEEPSEEK_API_KEY?.trim();
    if (!key) throw new Error('AI demo is not configured');

    const response = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
        signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]),
        body: JSON.stringify({
            model: 'deepseek-chat',
            max_tokens: 96,
            response_format: { type: 'json_object' },
            messages: [
                {
                    role: 'system',
                    content: 'Return JSON. The only editable property is the emissive panel brightness. Output {"operation":"setEmissionScale","scale":number}, where scale is an absolute number between 0 and 3. Use the current scale for relative changes. Return {} for other objects, unsupported operations, or unclear requests.',
                },
                { role: 'user', content: JSON.stringify({ prompt, current }) },
            ],
        }),
    });
    if (response.status === 402 || response.status === 429) throw new Error('AI demo quota is temporarily unavailable');
    if (!response.ok) throw new Error('AI request failed, please try again later');
    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;
    if (typeof content !== 'string') return null;
    try {
        return readEmissionCommand(JSON.parse(content));
    } catch {
        return null;
    }
}
