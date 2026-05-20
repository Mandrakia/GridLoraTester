// See https://kit.svelte.dev/docs/types#app for info
declare global {
    namespace App {
        // interface Error {}
        interface Locals {
            /** Set by hooks.server.ts: true when the request is authenticated
             * (or when no GLT_PASSWORD gate is configured). */
            authed: boolean;
        }
        // interface PageData {}
        // interface PageState {}
        // interface Platform {}
    }
}

export {};
