import React from "react";

export default function FileDropZone({ dragging, onDragOver, onDragEnter, onDragLeave, onDrop, onClick, label, children }: {
  dragging: boolean;
  onDragOver: React.DragEventHandler;
  onDragEnter: React.DragEventHandler;
  onDragLeave: React.DragEventHandler;
  onDrop: React.DragEventHandler;
  onClick: () => void;
  label: string;
  children?: React.ReactNode;
}) {
  return (
    <div
      onClick={onClick}
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed py-12 cursor-pointer transition-colors
        ${dragging ? "border-[#1E6B70] bg-[#1E6B70]/10" : "border-[#1D3335]/30 bg-white/40 hover:bg-white/60"}`}
    >
      <span className="text-3xl">↑</span>
      <p className="text-sm text-[#1D3335] text-center">{label}</p>
      {children}
    </div>
  );
}