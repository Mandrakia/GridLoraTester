import type { Actions, PageServerLoad } from './$types';
import { fail } from '@sveltejs/kit';
import {
    createPromptSet,
    deletePromptSet,
    listPromptSets,
    parsePromptsBlob,
    updatePromptSet
} from '$lib/server/prompts';

export const load: PageServerLoad = () => {
    return { prompt_sets: listPromptSets() };
};

function readFields(data: FormData) {
    const name = String(data.get('name') ?? '').trim();
    const blob = String(data.get('prompts') ?? '');
    return { name, prompts: parsePromptsBlob(blob), blob };
}

export const actions: Actions = {
    create: async ({ request }) => {
        const data = await request.formData();
        const { name, prompts, blob } = readFields(data);
        if (!name) return fail(400, { error: 'Name is required', name, blob });
        if (prompts.length === 0)
            return fail(400, { error: 'At least one prompt is required', name, blob });
        try {
            createPromptSet(name, prompts);
            return { ok: true };
        } catch (e) {
            return fail(500, { error: (e as Error).message, name, blob });
        }
    },

    update: async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        const { name, prompts, blob } = readFields(data);
        if (!name) return fail(400, { error: 'Name is required', id, name, blob });
        if (prompts.length === 0)
            return fail(400, { error: 'At least one prompt is required', id, name, blob });
        try {
            updatePromptSet(id, name, prompts);
            return { ok: true };
        } catch (e) {
            return fail(500, { error: (e as Error).message });
        }
    },

    delete: async ({ request }) => {
        const data = await request.formData();
        const id = Number(data.get('id'));
        if (!Number.isFinite(id) || id <= 0) return fail(400, { error: 'Bad id' });
        deletePromptSet(id);
        return { ok: true };
    }
};
