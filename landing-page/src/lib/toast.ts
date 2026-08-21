export type ToastVariant = "success" | "error" | "info" | "warning";

export interface Toast {
  id: number;
  message: string;
  variant: ToastVariant;
  duration: number;
  action?: ToastAction;
}

export interface ToastAction {
  label: string;
  onClick: () => void;
}

type Listener = () => void;

let toasts: Toast[] = [];
const listeners = new Set<Listener>();
let nextId = 1;

const DEFAULT_DURATION = 4000;

function emitChange() {
  for (const listener of listeners) listener();
}

function addToast(
  message: string,
  variant: ToastVariant,
  opts?: { duration?: number; action?: ToastAction },
) {
  const id = nextId++;
  const resolvedDuration = opts?.duration ?? DEFAULT_DURATION;
  toasts = [
    ...toasts,
    { id, message, variant, duration: resolvedDuration, action: opts?.action },
  ];
  emitChange();
  if (resolvedDuration > 0)
    setTimeout(() => dismissToast(id), resolvedDuration);
  return id;
}

export function dismissToast(id: number) {
  const next = toasts.filter((t) => t.id !== id);
  if (next.length === toasts.length) return;
  toasts = next;
  emitChange();
}

export function clearToasts(): void {
  if (toasts.length === 0) return;
  toasts = [];
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
  success: (
    message: string,
    opts?: { duration?: number; action?: ToastAction },
  ) => addToast(message, "success", opts),
  error: (
    message: string,
    opts?: { duration?: number; action?: ToastAction },
  ) => addToast(message, "error", opts),
  info: (message: string, opts?: { duration?: number; action?: ToastAction }) =>
    addToast(message, "info", opts),
  warning: (
    message: string,
    opts?: { duration?: number; action?: ToastAction },
  ) => addToast(message, "warning", opts),
  dismiss: dismissToast,
  clear: clearToasts,
};
