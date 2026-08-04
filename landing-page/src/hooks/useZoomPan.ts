import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";

const MIN_SCALE = 1;
const MAX_SCALE = 6;
const ZOOM_STEP = 0.4;

interface ZoomPanState {
    scale: number;
    x: number;
    y: number;
}

const IDLE: ZoomPanState = { scale: MIN_SCALE, x: 0, y: 0 };

/** Clamp scale to [MIN_SCALE, MAX_SCALE] and snap back to `IDLE` (centered,
 * unpanned) once zoomed all the way back out, so panning can't get "stuck"
 * offset at 1x zoom. */
function clampState(s: ZoomPanState): ZoomPanState {
    const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, s.scale))
    return scale === MIN_SCALE ? IDLE : { ...s, scale };
}

/**
 * Wheel-zoom + drag-to-pan for content rendered inside `containerRef`.
 * Apply `transformStyle` to a wrapper div around the zoomable content (e.g.
 * an <img> + its overlay <canvas>) — scaling via CSS transform keeps
 * `clientWidth`/`clientHeight`-based overlay coordinate math correct at any
 * zoom level, so overlays never need to redraw on zoom, only on image load.
 */
export function useZoomPan() {
    // A state-backed callback ref, not a plain useRef: the zoom container
    // only enters the DOM once its parent's async data finishes loading, and
    // a plain ref wouldn't re-trigger the wheel-listener effect below when
    // that happens later — the effect would've already run once (finding
    // `.current` null) and never fire again since its deps never change.
    const [containerNode, setContainerNode] = useState<HTMLDivElement | null>(null);
    const containerRef = useCallback((node: HTMLDivElement | null) => {
        setContainerNode(node);
    }, []);
    const [state, setState] = useState<ZoomPanState>(IDLE);
    const [dragging, setDragging] = useState(false);
    const dragOrigin = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(
    null,
    );

    const zoomBy = useCallback((delta: number, origin?: { x: number; y: number }) => {
        setState((prev) => {
            const nextScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, prev.scale + delta));
            if (nextScale === prev.scale) return prev;
            if (!origin || !containerNode) {
                return clampState({ ...prev, scale: nextScale });
            }
            // Keep the point under the cursor stationary while zooming.
            const rect = containerNode.getBoundingClientRect();
            const cx = origin.x - rect.left - rect.width / 2;
            const cy = origin.y - rect.top - rect.height / 2;
            const ratio = nextScale / prev.scale;
            return clampState({
                scale: nextScale,
                x: cx - (cx - prev.x) * ratio,
                y: cy - (cy - prev.y) * ratio,
            });
            });
    }, [containerNode]);

    // Native, non-passive listener: React's synthetic onWheel is passive by
    // default, so e.preventDefault() inside it is silently ignored (and logs a
    // warning) — the page would scroll instead of zooming.
    useEffect(() => {
        if (!containerNode) return;
        const onWheel = (e: WheelEvent) => {
            e.preventDefault();
            zoomBy(e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP, { x: e.clientX, y: e.clientY });
        };
        containerNode.addEventListener("wheel", onWheel, { passive: false });
        return () => containerNode.removeEventListener("wheel", onWheel);
    }, [containerNode, zoomBy]);

    const onMouseDown = useCallback(
        (e: MouseEvent<HTMLDivElement>) => {
            if (state.scale <= MIN_SCALE) return;
            dragOrigin.current = { startX: e.clientX, startY: e.clientY, origX: state.x, origY: state.y};
            setDragging(true);
        },
        [state],
    );

    const onMouseMove = useCallback((e: MouseEvent<HTMLDivElement>) => {
        if (!dragOrigin.current) return;
        const { startX, startY, origX, origY } = dragOrigin.current;
        setState((prev) => 
            clampState({ ...prev, x: origX + (e.clientX - startX), y: origY + (e.clientY - startY) }),
        );
    }, []);

    const endDrag = useCallback(() => {
        dragOrigin.current = null;
        setDragging(false);
    }, []);

    const reset = useCallback(() => setState(IDLE), []);
    const zoomIn = useCallback(() => zoomBy(ZOOM_STEP), [zoomBy]);
    const zoomOut = useCallback(() => zoomBy(-ZOOM_STEP), [zoomBy]);

    return {
        containerRef,
        scale: state.scale,
        isPannable: state.scale > MIN_SCALE,
        isDragging: dragging,
        canZoomIn: state.scale < MAX_SCALE,
        canZoomOut: state.scale > MIN_SCALE,
        transformStyle: {
            transform: `translate(${state.x}px, ${state.y}px) scale(${state.scale})`,
            transformOrigin: "center center",
            transition: dragging ? "none" : "transform 0.05s linear",
        },
        panHandlers: {
            onMouseDown,
            onMouseMove,
            onMouseUp: endDrag,
            onMouseLeave: endDrag,
        },
        zoomIn,
        zoomOut,
        reset,
    };
}