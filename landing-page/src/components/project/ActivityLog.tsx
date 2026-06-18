import { useEffect, useState } from "react";
import { authHeaders } from "../../hooks/useAuth";
import { formatActivity, formatRelativeTime } from "../../utils/time";
import type { ActivityEntry } from "../../utils/time";

export default function ActivityLog({ projectId }: { projectId: number }) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<ActivityEntry[]>([]);

  useEffect(() => {
    if (!open) return;
    fetch(`/api/projects/${projectId}/activity`, { headers: authHeaders() })
      .then(r => r.json())
      .then(setEntries);
  }, [open, projectId]);

  return (
    <div className="w-48 bg-[#C8E6E3]/30 rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full px-5 py-3 flex items-center justify-between text-white/60 hover:text-white text-xs cursor-pointer transition-colors"
      >
        <span>activity log</span>
        <span className={`transition-transform ${open ? "rotate-180" : ""}`}>▾</span>
      </button>
      {open && (
        <div className="px-4 pb-4 flex flex-col gap-3 max-h-64 overflow-y-auto">
          {entries.length === 0
            ? <p className="text-xs text-white/40">no activity yet</p>
            : entries.map((e, i) => (
                <div key={i}>
                  <p className="text-xs text-white/80 leading-snug">{formatActivity(e)}</p>
                  <p className="text-[10px] text-white/40 mt-0.5">{formatRelativeTime(e.createdAt)}</p>
                </div>
              ))
          }
        </div>
      )}
    </div>
  );
}
