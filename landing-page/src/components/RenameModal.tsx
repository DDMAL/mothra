interface RenameModalProps {
    label: string;
    value: string;
    onChange: (v: string) => void;
    onSubmit: () => void;
    onClose: () => void;
}

export default function RenameModal({ label, value, onChange, onSubmit, onClose }: RenameModalProps) {
    return (
        <>
            <div className="fixed inset-0 z-40" onClick={onClose} />
            <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 relative shadow-2xl">
                <button
                onClick={onClose}
                className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
                >
                ✕
                </button>
                <h2 className="text-xl text-[#1D3335] text-center">rename {label}</h2>
                <input
                autoFocus
                value={value}
                onChange={(e) => onChange(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") onSubmit(); }}
                className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm"
                />
                <button
                onClick={onSubmit}
                className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
                >
                rename {label}
                </button>
            </div>
        </>
    );
}