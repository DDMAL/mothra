import { useEffect } from "react";
import type { AuthStatus, View } from "../types";

// `authStatus` is a dependency only so this re-runs when it changes: while a
// stored session is being checked, AppRouter renders a placeholder on
// "landing" instead of the hero, so if the check fails the hero mounts later
// with `view` unchanged -- and its .hero-fade elements would never be found.
export function useScrollFade(view: View, authStatus: AuthStatus) {
  useEffect(() => {
    if (view !== "landing") {
      document
        .querySelectorAll(".fade-target")
        .forEach((el) => el.classList.add("visible"));
      return;
    }

    let timer: ReturnType<typeof setTimeout> | undefined;
    if (view === "landing") {
      const heroTargets = document.querySelectorAll(".hero-fade");
      timer = setTimeout(() => {
        heroTargets.forEach((el) => el.classList.add("visible"));
      }, 100);
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 },
    );

    document
      .querySelectorAll(".scroll-fade")
      .forEach((el) => observer.observe(el));

    return () => {
      if (timer) clearTimeout(timer);
      observer.disconnect();
    };
  }, [view, authStatus]);
}
