import { useSyncExternalStore } from "react";
import {
    subscribeToasts,
    getToastSnapshot,
    dismissToast,
    type ToastVariant,
} from "../../lib/toast";

const VARIANT_STYLES: Record<ToastVariant, string> = {
    info: "bg-[#1D3335] text-white",
    success: "bg-[#1E6B70] text-white",
    error: "bg-red-900 text-red-200",
    warning: "bg-[#1D3335] text-amber-300",
};

export default function ToastContainer() {
    const toasts = useSyncExternalStore(subscribeToasts, getToastSnapshot);

    if (toasts.length === 0) return null;

    return (
        <div className="fixed top-14 inset-x-0 z-[60] flex flex-col items-center gap-2 pointer-events-none">
            {toasts.map((t) => (
                <div
                    key={t.id}
                    className={`animate-fade-in pointer-events-auto flex items-center gap-3 text-sm px-5 py-2.5 rounded-2xl shadow-2xl ${VARIANT_STYLES[t.variant]}`}
                >
                    <span>{t.message}</span>
                    <button
                        onClick={() => dismissToast(t.id)}
                        className="text-white/60 hover:text-white cursor-pointer text-base leading-none"
                    >
                        ✕
                    </button>
                </div>
            ))}
        </div>
    );
}