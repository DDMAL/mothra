import { authHeaders, setToken, clearToken } from "../hooks/useAuth";

// replaces raw fetch() for all /api/* calls
let onUnauthenticated: (() => void) | null = null;

export function registerUnauthenticatedHandler(fn: () => void) {
    onUnauthenticated = fn;
}

export async function apiFetch(
    input: RequestInfo,
    init?: RequestInit,
): Promise<Response> {
    const resp = await fetch(input, {
        ...init,
        headers: { ...authHeaders(), ...(init?.headers ?? {}) },
    });

    if (resp.status !== 401) return resp;

    // attempt a silent refresh
    const refresh = await fetch("/api/auth/refresh", { headers: authHeaders() });
    if (!refresh.ok) {
        onUnauthenticated?.();
        return resp;
    }
    const {access_token } = await refresh.json();
    setToken(access_token);

    // retry original request once w/ new token
    return fetch(input, {
        ...init,
        headers: { Authorization: `Bearer ${access_token}`, ...(init?.headers ?? {}) },
    });
}