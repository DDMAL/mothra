import { useEffect, useState } from "react";
import { apiFetch } from "../../lib/apiFetch";

interface AuthImageProps {
  src: string;
  alt?: string;
  className?: string;
  onLoad?: (e: React.SyntheticEvent<HTMLImageElement>) => void;
}

export function AuthImage({
  src,
  alt = "",
  className,
  onLoad,
}: AuthImageProps) {
  const [blobSrc, setBlobSrc] = useState<string | null>(null);
  useEffect(() => {
    let revoked = false;
    let objectUrl: string | null = null;
    apiFetch(src)
      .then((r) => r.blob())
      .then((blob) => {
        if (revoked) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobSrc(objectUrl);
      });
    return () => {
      revoked = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);
  if (!blobSrc) return null;
  return <img src={blobSrc} alt={alt} className={className} onLoad={onLoad} />;
}
