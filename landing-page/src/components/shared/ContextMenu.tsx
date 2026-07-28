import { useLayoutEffect, useRef, useState } from "react";

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
  const [pos, setPos] = useState({ top: y + 8, left: x - 80 });

  useLayoutEffect(() => {
    const el = menuRef.current;
    if (!el) return;
    const { height, width } = el.getBoundingClientRect();

    let top = y + 8;
    if (top + height > window.innerHeight - MARGIN) {
      top = y - height - 8;
    }
    top = Math.min(Math.max(top, MARGIN), Math.max(MARGIN, window.innerHeight - height - MARGIN));

    let left = x - 80;
    left = Math.min(Math.max(left, MARGIN), Math.max(MARGIN, window.innerWidth - width - MARGIN));

    setPos({ top, left });
  }, [x, y]);

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
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
    </>
  );
}
