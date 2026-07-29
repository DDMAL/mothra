export const MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024;
export const TARGET_RESIZE_BYTES = 2 * 1024 * 1024;

export function getOversizedFiles(
    files: File[],
    thresholdBytes: number = MAX_IMAGE_SIZE_BYTES,
) : File[] {
    return files.filter((f) => f.size > thresholdBytes);
}

// Downscales/re-encodes an image file as JPEG until it's under targetBytes
// (or a bounded number of attempts is exhausted - some pathological images
// may not compress below the target, so this is best-effort, not guaranteed).
export async function resizeImageFile(
    file: File,
    targetBytes: number = TARGET_RESIZE_BYTES,
): Promise<File> {
    const bitmap = await createImageBitmap(file);
    // area scales roughly with byte size for a given quality, so this gives a
    // reasonable starting point; the loop below corrects if it's still too big
    let scale = Math.min(1, Math.sqrt(targetBytes / file.size));
    let quality = 0.9;
    let blob: Blob | null = null;

    for (let attempt = 0; attempt < 6; attempt++) {
        const width = Math.max(1, Math.round(bitmap.width * scale));
        const height = Math.max(1, Math.round(bitmap.height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d")!.drawImage(bitmap, 0, 0, width, height);
        blob = await new Promise<Blob>((resolve) =>
            canvas.toBlob((b) => resolve(b!), "image/jpeg", quality),
        );
        if (blob.size <= targetBytes) break;
        if (quality > 0.5) {
            quality -= 0.15;
        } else {
            scale *=0.8;
        }
    }
    bitmap.close();

    const newName = file.name.replace(/\.[^./\\]+$/, "") + ".jpg";
  return new File([blob!], newName, { type: "image/jpeg" });
}