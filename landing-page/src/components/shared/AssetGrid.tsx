import type { ReactNode } from "react";
import { Fragment } from "react";
import type React from "react";
import Paginator from "./Paginator";

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
  totalPages: number;
  renderThumbnail: (item: T) => ReactNode;
  getItemBadge?: (name: string) => string | null;
  groupBy?: (item: T) => string;
}

export default function AssetGrid<T extends AssetItem>({
  pagedItems,
  pageOffset,
  section,
  usedNames,
  totalPages,
  renderThumbnail,
  getItemBadge,
  groupBy,
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
          const used = usedNames.includes(item.name);
          const badge = used ? (getItemBadge?.(item.name) ?? null) : null;
          const group = groupBy?.(item);
          const prevGroup = groupBy && pageIdx > 0 ? groupBy(pagedItems[pageIdx - 1]) : undefined;
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
                  className={`relative aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer transition-shadow flex items-center justify-center
                          ${section.selectedIds.has(item.id) ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}
                          ${used ? "opacity-40 cursor-default" : ""}`}
                  onClick={(e) => {
                    if (!used) section.handleClick(e, item.id, idx);
                  }}
                >
                  {renderThumbnail(item)}
                  {badge && (
                    <div className="absolute bottom-0 inset-x-0 flex justify-center pb-1.5 pointer-events-none">
                      <span className="bg-[#1D3335]/80 text-white text-xs px-2 py-0.5 rounded-full">
                        {badge}
                      </span>
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-between gap-1">
                  <span className={`text-sm text-white truncate ${used ? "opacity-40" : ""}`}>
                    {item.name}
                  </span>
                  {!used && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        section.setMenu({ id: item.id, x: e.clientX, y: e.clientY });
                      }}
                      className="text-white text-lg leading-none hover:opacity-70 cursor-pointer flex-shrink-0"
                    >
                      ⋮
                    </button>
                  )}
                </div>
              </div>
            </Fragment>
          );
        })}
      </div>
      {totalPages > 1 && (
        <Paginator page={section.page} totalPages={totalPages} onPageChange={section.setPage} />
      )}
    </>
  );
}
