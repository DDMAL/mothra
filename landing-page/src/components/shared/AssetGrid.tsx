import type { ReactNode } from "react";
import { Fragment } from "react";
import type React from "react";
import Paginator from "./Paginator";
import TruncatedName from "./TruncatedName";

interface AssetItem {
  id: string;
  name: string;
}

interface AssetSection {
  selectedIds: Set<string>;
  handleClick: (e: React.MouseEvent, id: string, idx: number) => void;
  setMenu: (v: { id: string; x: number; y: number } | null) => void;
  page: number;
  setPage: React.Dispatch<React.SetStateAction<number>>;
}

interface AssetGridProps<T extends AssetItem> {
  pagedItems: T[];
  pageOffset: number;
  section: AssetSection;
  usedNames: string[];
  // Which item field `usedNames` matches against. Defaults to "name"
  // (models/MEI files have no duplicate-name concern to fix). mothra#241
  // follow-up (CodeRabbit): ImageTab.tsx passes "id" instead, since
  // duplicate-named image uploads need to be matched/selected/removed
  // independently, which a name match can't do.
  usedKey?: "name" | "id";
  totalPages: number;
  renderThumbnail: (item: T) => ReactNode;
  getItemBadge?: (item: T) => string | null;
  groupBy?: (item: T) => string;
  onUse?: (item: T) => void;
  // mothra#247: symmetric counterpart to onUse -- lets an already-used item
  // be deselected directly from the grid instead of only via the project
  // page's "selected:" side list.
  onRemove?: (item: T) => void;
  topLeftBadge?: (item: T) => ReactNode;
}

export default function AssetGrid<T extends AssetItem>({
  pagedItems,
  pageOffset,
  section,
  usedNames,
  usedKey = "name",
  totalPages,
  renderThumbnail,
  getItemBadge,
  groupBy,
  onUse,
  onRemove,
  topLeftBadge,
}: AssetGridProps<T>) {
  return (
    <>
      <div
        className="grid grid-cols-5 gap-4"
        onMouseDown={(e) => {
          if (e.shiftKey) e.preventDefault();
        }}
      >
        {pagedItems.map((item, pageIdx) => {
          const idx = pageOffset + pageIdx;
          const used = usedNames.includes(item[usedKey]);
          const badge = used ? (getItemBadge?.(item) ?? null) : null;
          const group = groupBy?.(item);
          const prevGroup =
            groupBy && pageIdx > 0
              ? groupBy(pagedItems[pageIdx - 1])
              : undefined;
          const showHeader = groupBy && group !== prevGroup;
          return (
            <Fragment key={item.id}>
              {showHeader && (
                <div className="col-span-5 text-white/70 text-xs font-mono uppercase tracking-wide mt-4 first:mt-0 pb-1 border-b border-white/20">
                  {group}
                </div>
              )}
              <div className="flex flex-col gap-2">
                <div
                  className={`group relative aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer transition-shadow flex items-center justify-center
                          ${section.selectedIds.has(item.id) ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}
                          ${used ? "opacity-40" : ""}`}
                  onClick={(e) => {
                    // mothra#247: a used item is only selectable when the
                    // caller wired up onRemove (today, just ImageTab.tsx's
                    // "remove N from selection" bulk action) -- scoped this
                    // way rather than relaxing the guard for every AssetGrid
                    // consumer, since MeiTab.tsx/ModelTab.tsx have no
                    // deselect concept and their existing multi-select
                    // "delete N" action should not suddenly gain the
                    // ability to bulk-delete already-used models/mei files
                    // that were previously unselectable.
                    if (!used || onRemove) section.handleClick(e, item.id, idx);
                  }}
                >
                  {renderThumbnail(item)}
                  {(topLeftBadge?.(item) ||
                    (onUse && !used) ||
                    (onRemove && used)) && (
                    <div className="absolute top-1.5 left-1.5 z-20 flex items-center gap-1">
                      {topLeftBadge?.(item)}
                      {onUse && !used && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onUse(item);
                          }}
                          className="px-1.5 py-0.5 bg-black/40 text-white text-[9px] font-mono rounded hover:bg-black/70 cursor-pointer"
                        >
                          use
                        </button>
                      )}
                      {onRemove && used && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onRemove(item);
                          }}
                          title="Remove from selection"
                          className="px-1.5 py-0.5 bg-black/40 text-white text-[9px] font-mono rounded hover:bg-black/70 cursor-pointer"
                        >
                          remove
                        </button>
                      )}
                    </div>
                  )}
                  {!used && (
                    <button
                      aria-label={`Open actions for ${item.name}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        section.setMenu({
                          id: item.id,
                          x: e.clientX,
                          y: e.clientY,
                        });
                      }}
                      // Faintly visible by default (not opacity-0) so it stays
                      // discoverable on touch/keyboard, which have no hover —
                      // hovering the thumbnail just makes it more prominent.
                      className="absolute top-1.5 right-1.5 z-20 w-6 h-6 flex items-center justify-center rounded-full bg-black/40 text-white text-base leading-none opacity-70 group-hover:opacity-100 group-hover:bg-black/70 hover:opacity-100 hover:bg-black/70 transition-all cursor-pointer"
                    >
                      ⋮
                    </button>
                  )}
                  {badge && (
                    <div className="absolute bottom-0 inset-x-0 flex justify-center pb-1.5 pointer-events-none">
                      <span className="bg-[#1D3335]/80 text-white text-xs px-2 py-0.5 rounded-full">
                        {badge}
                      </span>
                    </div>
                  )}
                </div>
                <TruncatedName
                  name={item.name}
                  className={`text-sm text-white ${used ? "opacity-40" : ""}`}
                />
              </div>
            </Fragment>
          );
        })}
      </div>
      {totalPages > 1 && (
        <Paginator
          page={section.page}
          totalPages={totalPages}
          onPageChange={section.setPage}
        />
      )}
    </>
  );
}
