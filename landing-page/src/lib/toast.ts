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

// Drops any lingering *persistent* toasts (duration: 0) -- used to clear a
// job-status toast (see useActiveJobWatcher in App.tsx) once the user
// navigates away from wherever they were when it appeared, since those are
// deliberately given no auto-dismiss timer so a background job's
// completion isn't missed regardless of page. Scoped to duration:0 only:
// an ordinary self-dismissing toast (e.g. apiFetch's session-expiry notice,
// fired immediately before the logout that navigates to "landing") already
// has its own timer and must survive a same-tick navigation long enough to
// actually be read, not vanish the instant the view changes.
export function clearToasts(): void {
  const next = toasts.filter((t) => t.duration !== 0);
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
