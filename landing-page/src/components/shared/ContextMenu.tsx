import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface ContextMenuItem {
  label: string;
  onClick: () => void;
}
interface ContextMenuProps {
  x: number;
  y: number;
  items: ContextMenuItem[];
  onClose: () => void;
}

const MARGIN = 8;

export default function ContextMenu({
  x,
  y,
  items,
  onClose,
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  // page (document) coordinates, not viewport coordinates, so the menu
  // scrolls together with the 3-dot button that opened it instead of
  // staying glued to a fixed spot on screen.
  const [pos, setPos] = useState({ top: y + window.scrollY + 8, left: x + window.scrollX - 80 });

  useLayoutEffect(() => {
    const el = menuRef.current;
    if (!el) return;
    const { height, width } = el.getBoundingClientRect();

    // clamp/flip against the viewport as it is right now (x/y are
    // viewport-relative click coordinates), then convert to page
    // coordinates for the actual `absolute`-positioned style below.
    let top = y + 8;
    if (top + height > window.innerHeight - MARGIN) {
      top = y - height - 8;
    }
    top = Math.min(Math.max(top, MARGIN), Math.max(MARGIN, window.innerHeight - height - MARGIN));

    let left = x - 80;
    left = Math.min(Math.max(left, MARGIN), Math.max(MARGIN, window.innerWidth - width - MARGIN));

    setPos({ top: top + window.scrollY, left: left + window.scrollX });
  }, [x, y]);

  return createPortal(
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        ref={menuRef}
        className="absolute z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
        style={{ top: pos.top, left: pos.left }}
      >
        {items.map((item) => (
          <button
            key={item.label}
            className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            onClick={item.onClick}
          >
            {item.label}
          </button>
        ))}
      </div>
    </>,
    document.body,
  );
}
