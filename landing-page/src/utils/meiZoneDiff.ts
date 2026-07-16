interface ZoneRect { id: string; ulx: number; uly: number; lrx: number; lry: number; }

export interface ZoneDiff {
    added: ZoneRect[]; // green
    removed: ZoneRect[]; // red
    moved: ZoneRect[]; // yellow
    unchanged: ZoneRect[]; // teal
}

function parseZones(xml: string): Map<string, ZoneRect> {
    const doc = new DOMParser().parseFromString(xml, "application/xml")
    const map = new Map<string, ZoneRect>();
    doc.querySelectorAll("zone").forEach(el => {
        const id = el.getAttributeNS("http://www.w3.org/XML/1998/namespace", "id")
            ?? el.getAttribute("xml:id") ?? "";
        if (!id) return;
        map.set(id, {
            id,
            ulx: Number(el.getAttribute("ulx") ?? 0),
            uly: Number(el.getAttribute("uly") ?? 0),
            lrx: Number(el.getAttribute("lrx") ?? 0),
            lry: Number(el.getAttribute("lry") ?? 0),
        });
    });
    return map;
}

export function diffZones(originalXml: string, correctedXml: string): ZoneDiff {
    const orig = parseZones(originalXml)
    const corr = parseZones(correctedXml)
    const added: ZoneRect[] = [], removed: ZoneRect[] = [], moved: ZoneRect[] = [], unchanged: ZoneRect[] = [];
    
    corr.forEach((zone, id) => {
        if (!orig.has(id)) added.push(zone);
    });
    orig.forEach((origZone, id) => {
        const corrZone = corr.get(id);
        if (!corrZone) { removed.push(origZone); return; }
        const same = origZone.ulx === corrZone.ulx && origZone.uly === corrZone.uly
              && origZone.lrx === corrZone.lrx && origZone.lry === corrZone.lry;
        if (!same) moved.push(corrZone); // show corrected position
        else unchanged.push(corrZone);
    });
    return { added, removed, moved, unchanged };
}
