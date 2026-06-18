export default function QuickLookModal({ onClose, children }: {
  onClose: () => void; children: React.ReactNode
}) {
  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/60" onClick={onClose} />
      <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
        <div className="relative bg-[#1D3335] rounded-2xl shadow-2xl p-6 flex flex-col gap-4 max-w-2xl w-full mx-4 pointer-events-auto animate-fade-in">
          <button onClick={onClose}
            className="absolute top-3 right-4 text-white/60 hover:text-white text-2xl leading-none cursor-pointer">×</button>
            {children}
        </div>
      </div>
    </>
  );
}