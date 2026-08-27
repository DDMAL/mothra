import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { TutorialStep } from "./tutorialSteps";

// Matches Navbar.tsx's `sticky top-0 z-50 h-14` -- nothing this overlay
// draws (scrim, spotlight ring, or callout box) should cover the nav bar,
// even though the overlay's own z-index (below) sits above it.
const NAVBAR_HEIGHT_PX = 56;

interface TutorialOverlayProps {
    step: TutorialStep;
    onNext: () => void;
    onSkip: () => void;
}

export default function TutorialOverlay({ step, onNext, onSkip }: TutorialOverlayProps) {
    const [rect, setRect] = useState<DOMRect | null>(null);

    useEffect(() => {
        const measure = () => {
            if (!step.target) {
                setRect(null);
                return;
            }
            const el = document.querySelector(`[data-tutorial-target="${step.target}"]`);
            setRect(el ? el.getBoundingClientRect() : null);
        };
        measure();
        window.addEventListener("resize", measure);
        window.addEventListener("scroll", measure, true);
        return () => {
            window.removeEventListener("resize", measure);
            window.removeEventListener("scroll", measure, true);
        };
    }, [step.target]);

    const isAction = step.type === "action";

    return createPortal(
        <>
        {/* Informational steps dim the page (nothing to click through to);
            action steps render no scrim at all, so the real "begin" button
            stays fully interactive underneath -- see the plan's "Known
            simplifications" re: no true spotlight-cutout/click-blocking.
            Starts below the nav bar (top: NAVBAR_HEIGHT_PX, not inset-0) so
            it never dims/covers it. */}
        {!isAction && (
            <div
            className="fixed inset-x-0 bottom-0 z-[60] bg-black/60"
            style={{ top: NAVBAR_HEIGHT_PX }}
            />
        )}
        {rect && (
            <div
            className="fixed z-[61] rounded-xl ring-4 ring-[#1E6B70] pointer-events-none transition-all"
            style={{
                // Floored at the nav bar -- a target whose rect starts at or
                // above it (shouldn't happen with today's targets, all well
                // below it, but kept as a hard guarantee) never draws the
                // ring over the nav bar.
                top: Math.max(rect.top - 6, NAVBAR_HEIGHT_PX),
                left: rect.left - 6,
                width: rect.width + 12,
                height: rect.height + 12,
            }}
            />
        )}
        <div
            className="fixed z-[62] bg-[#C8E6E3] rounded-2xl p-5 max-w-sm shadow-2xl flex flex-col gap-3"
            style={
            rect
                ? {
                    top: Math.max(
                        Math.min(rect.bottom + 16, window.innerHeight - 200),
                        NAVBAR_HEIGHT_PX + 8,
                    ),
                    left: Math.max(16, Math.min(rect.left, window.innerWidth - 400)),
                }
                : { bottom: 32, left: "50%", transform: "translateX(-50%)" }
            }
        >
            <h3 className="text-lg font-bold text-[#1D3335]">{step.title}</h3>
            <p className="text-sm text-[#1D3335]/80">{step.body}</p>
            <div className="flex justify-between items-center mt-1">
            {step.phase !== "done" && (
                <button
                onClick={onSkip}
                className="text-xs text-[#1D3335]/60 hover:text-[#1D3335] cursor-pointer"
                >
                skip tour
                </button>
            )}
            {isAction ? (
                <span className="text-xs text-[#1D3335]/60 italic">waiting for you...</span>
            ) : (
                // "done" has no further step to advance to -- its button
                // finishes (dismisses) the tour instead, which is also what
                // makes a "restart tutorial" control reappear afterward
                // (see useTutorialFlow.ts's canStart).
                <button
                onClick={step.phase === "done" ? onSkip : onNext}
                className="px-4 py-1.5 bg-[#1E6B70] text-white text-sm font-semibold rounded-full hover:opacity-90 cursor-pointer ml-auto"
                >
                {step.phase === "done" ? "finish" : <>next &rarr;</>}
                </button>
            )}
            </div>
        </div>
        </>,
        document.body,
    );
}