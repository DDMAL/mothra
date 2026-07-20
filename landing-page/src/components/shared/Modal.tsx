import type { ReactNode } from "react";

type ModalSize = "sm" | "lg" | "2xl" | "4xl" | "5xl";
type ModalBackdrop = "none" | "dim" | "dark";

interface ModalProps {
  onClose?: () => void;
  showCloseButton?: boolean;
  size?: ModalSize;
  backdrop?: ModalBackdrop;
  children: ReactNode;
}

const SIZE = {
  sm: "max-w-sm",
  lg: "max-w-lg",
  "2xl": "max-w-2xl",
  "4xl": "max-w-4xl",
  "5xl": "max-w-5xl",
};
const BACKDROP = { none: "", dim: "bg-black/30", dark: "bg-black/60" };

export default function Modal({
  onClose,
  showCloseButton = true,
  size = "lg",
  backdrop = "none",
  children,
}: ModalProps) {
  return (
    <>
      <div
        className={`fixed inset-0 z-40 ${BACKDROP[backdrop]}`}
        onClick={onClose}
      />
      <div
        className={`animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full ${SIZE[size]} bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 relative shadow-2xl`}
      >
        {onClose && showCloseButton && (
          <button
            onClick={onClose}
            className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
          >
            ✕
          </button>
        )}
        {children}
      </div>
    </>
  );
}
