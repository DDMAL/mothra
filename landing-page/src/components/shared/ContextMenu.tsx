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

export default function ContextMenu({
  x,
  y,
  items,
  onClose,
}: ContextMenuProps) {
  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
        style={{ top: y + 8, left: x - 80 }}
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
