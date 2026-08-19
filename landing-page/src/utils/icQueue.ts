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
  const imgResp = await apiFetch(`/api/images/${image.id}`);
  if (!imgResp.ok) throw new Error(`image fetch failed (${imgResp.status})`);
  const blob = await imgResp.blob();
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
  return buildEncodePair(image, data.xml_base64);
}
