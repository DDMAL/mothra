import { useEffect, useState } from "react";
import { authHeaders } from "../hooks/useAuth";

interface AuthImageProps {
    src: string;
    alt?: string;
    className?: string;
}

export function AuthImage({ src, alt = "", className }: AuthImageProps) {
    const [blobSrc, setBlobSrc] = useState<string | null>(null);
    useEffect(() => {
        let revoked = false;
        let objectUrl: string | null = null;
        fetch(src, { headers: authHeaders() })
            .then(r => r.blob())
            .then(blob => {
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
    return <img src={blobSrc} alt={alt} className={className} />;
}