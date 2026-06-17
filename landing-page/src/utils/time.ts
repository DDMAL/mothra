export function formatRelativeTime(ts: string): string {
    const hours = (Date.now() - new Date(ts).getTime()) / 3600000;
    if (hours < 1) return "< 1 hour ago";
    if (hours < 24) {
        const h = Math.floor(hours);
        return `${h}h ago`;
    }
    return new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function formatLastOpened(ts: string | undefined): string {
  if (!ts) return "never opened";
  const hours = (Date.now() - new Date(ts).getTime()) / 3600000;
  if (hours < 1) return "< 1 hour ago";
  if (hours < 24) {
    const h = Math.floor(hours);
    return `${h} hour${h !== 1 ? "s" : ""} ago`;
  }
  return new Date(ts).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}