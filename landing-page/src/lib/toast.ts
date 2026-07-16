export type ToastVariant = "success" | "error" | "info" | "warning";

export interface Toast {
    id: number;
    message: string;
    variant: ToastVariant;
    duration: number;
}

type Listener = () => void;

let toasts: Toast[] = [];
const listeners = new Set<Listener>();
let nextId = 1;

const DEFAULT_DURATION = 4000;

function emitChange() {
    for (const listener of listeners) listener();
}

function addToast(message: string, variant: ToastVariant, duration?: number) {
    const id = nextId++;
    const resolvedDuration = duration ?? DEFAULT_DURATION;
    toasts = [...toasts, { id, message, variant, duration: resolvedDuration }];
    emitChange();

    if (resolvedDuration > 0) {
        setTimeout(() => dismissToast(id), resolvedDuration);
    }

    return id;
}

export function dismissToast(id: number) {
    const next = toasts.filter((t) => t.id !== id);
    if (next.length === toasts.length) return;
    toasts = next;
    emitChange();
}

export function subscribeToasts(listener: Listener): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function getToastSnapshot(): Toast[] {
    return toasts;
}

export const toast = {
    success: (message: string, duration?: number) => addToast(message, "success", duration),
    error: (message: string, duration?: number) => addToast(message, "error", duration),
    info: (message: string, duration?: number) => addToast(message, "info", duration),
    warning: (message: string, duration?: number) => addToast(message, "warning", duration),
    dismiss: dismissToast,
}