interface PaginatorProps {
    page: number;
    totalPages: number;
    onPageChange: (updater: (p: number) => number) => void;
}

export default function Paginator({ page, totalPages, onPageChange }: PaginatorProps) {
    return (
        <div className="flex items-center justify-center gap-4 mt-6 text-white text-sm">
            <button onClick={() => onPageChange(p => p - 1)} disabled={page === 0} className="hover:opacity-70 disabled:opacity-30 cursor-pointer">←</button>
            <span>page {page + 1} of {totalPages}</span>
            <button onClick={() => onPageChange(p => p + 1)} disabled={page === totalPages - 1} className="hover:opacity-70 disabled:opacity-30 cursor-pointer">→</button>
        </div>
    );
}