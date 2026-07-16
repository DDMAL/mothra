import { useEffect, useState } from "react";
import type { MouseEvent } from "react";

export const ITEMS_PER_PAGE = 10;

interface HasId {
  id: string;
}

export function useAssetSection<T extends HasId>(items: T[]) {
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(
    null,
  );
  const [renameModal, setRenameModal] = useState<{ id: string } | null>(null);
  const [renameName, setRenameName] = useState("");
  const [uploadModal, setUploadModal] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [lastSelectedIdx, setLastSelectedIdx] = useState<number | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    const max = Math.max(0, Math.ceil(items.length / ITEMS_PER_PAGE) - 1);
    setPage((p) => Math.min(p, max));
  }, [items.length]);

  const handleClick = (e: MouseEvent, id: string, idx: number) => {
    e.stopPropagation();
    if (e.shiftKey) {
      e.preventDefault();
      setSelectedIds((prev) => {
        const next = new Set(prev);
        if (lastSelectedIdx !== null && lastSelectedIdx !== idx) {
          const lo = Math.min(lastSelectedIdx, idx);
          const hi = Math.max(lastSelectedIdx, idx);
          items.slice(lo, hi + 1).forEach((item) => next.add(item.id));
        } else {
          next.has(id) ? next.delete(id) : next.add(id);
        }
        return next;
      });
      setLastSelectedIdx(idx);
    } else {
      if (selectedIds.has(id)) {
        setSelectedIds((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      } else {
        setSelectedIds(new Set([id]));
        setLastSelectedIdx(idx);
      }
    }
  };

  const clearSelection = () => {
    setSelectedIds(new Set());
    setLastSelectedIdx(null);
  };

  return {
    menu,
    setMenu,
    renameModal,
    setRenameModal,
    renameName,
    setRenameName,
    uploadModal,
    setUploadModal,
    dragging,
    setDragging,
    selectedIds,
    setSelectedIds,
    lastSelectedIdx,
    setLastSelectedIdx,
    page,
    setPage,
    handleClick,
    clearSelection,
  };
}
