import { apiFetch } from "../lib/apiFetch";
import type { ProjectImage } from "../types";

/** The {xmlFile, imageFile} shape the encode-batch flow consumes. */
export interface EncodePair {
  xmlFile: File;
  imageFile: File;
  // mothra#241: the source project_images.id, threaded through to
  // tasks_encode.py so hint-resolution and the resulting mei_files row can
  // be matched by id instead of the not-necessarily-unique image name.
  imageId: string;
}

const stemOf = (name: string) => name.replace(/\.[^.]+$/, "");

/**
 * Per-page timing for the automatic IC pass, to the browser console.
 *
 * The auto pass is a strictly sequential loop of blocking round-trips
 * (IcAutoQueue's own loop, then this module's), and none of it goes through
 * the job-queue SSE stream that ProcessingPage renders — so unlike predict
 * and encode, there was no log anywhere showing where a slow pass spent its
 * time. The console is the right surface for it: it is per-page diagnostic
 * detail, not something the pipeline UI should grow a panel for.
 *
 * `detail` carries the backend's own breakdown when it sent one
 * (ic_api.py's `timing`: the /staging upload, the classify round, the
 * GameraXML export, and the size of the response the classify round
 * returns).
 */
function logTiming(
  imageName: string,
  label: string,
  detail: Record<string, unknown>,
) {
  const parts = Object.entries(detail)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(2) : v}`);
  console.info(`[timing] ic ${label} — ${imageName}: ${parts.join(", ")}`);
}

/**
 * Turn IC's GameraXML (base64) + a project image into an encode pair. Shared
 * by the interactive path (which builds its pairs when "encode batch" is
 * pressed, not when a page is queued — see InteractiveClassifier's
 * handleEncodeBatch) and the automatic queue-all path.
 */
export async function buildEncodePair(
  image: ProjectImage,
  xmlBase64: string,
): Promise<EncodePair> {
  const xmlBytes = Uint8Array.from(atob(xmlBase64), (c) => c.charCodeAt(0));
  const xmlFile = new File([xmlBytes], `${stemOf(image.name)}.xml`, {
    type: "application/xml",
  });
  const fetchStart = performance.now();
  const imgResp = await apiFetch(`/api/images/${image.id}`);
  if (!imgResp.ok) throw new Error(`image fetch failed (${imgResp.status})`);
  const blob = await imgResp.blob();
  logTiming(image.name, "image-refetch", {
    seconds: (performance.now() - fetchStart) / 1000,
    bytes: blob.size,
  });
  const imageFile = new File([blob], image.name, {
    type: blob.type || "image/png",
  });
  return { xmlFile, imageFile, imageId: image.id };
}

/**
 * Classify one page server-side with the shared training set (no IC iframe)
 * and return its encode pair. Requires a non-empty training set — classify
 * has no training pool without one.
 */
export async function autoQueueImage(
  projectId: number,
  image: ProjectImage,
  trainingPresets: string[],
  trainingFiles: File[],
): Promise<EncodePair> {
  const queueStart = performance.now();
  const form = new FormData();
  form.append("imageName", image.name);
  // CodeRabbit (ic_api.py#L254): image_name alone can't disambiguate a
  // duplicate-named upload -- send the real id so the backend resolves the
  // exact page instead of an arbitrary same-named match.
  form.append("imageId", image.id);
  if (trainingPresets.length > 0)
    form.append("training_presets", JSON.stringify(trainingPresets));
  trainingFiles.forEach((f) => form.append("training_files", f));
  const r = await apiFetch(`/api/projects/${projectId}/ic/auto-queue`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
  const data = await r.json();
  logTiming(image.name, "auto-queue", {
    seconds: (performance.now() - queueStart) / 1000,
    ...(data.timing ?? {}),
  });
  return buildEncodePair(image, data.xml_base64);
}
