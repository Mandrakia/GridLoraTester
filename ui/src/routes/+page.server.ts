// Root → datasets (the default entry point).
import { redirect } from '@sveltejs/kit';

export const load = () => {
    throw redirect(307, '/datasets');
};
