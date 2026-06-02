function scrollToCenter(id: string) {
  document
    .getElementById(id)
    ?.scrollIntoView({ behavior: "smooth", block: "center" });
}

export default function About() {
  return (
    <div className="flex-1 bg-[#4AADAA]">
      {/* Intro block */}
      <section className="scroll-fade fade-target px-6 pt-12 pb-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl sm:text-4xl font-bold italic text-white mb-6">
            about mothra
          </h1>
          <p className="text-[#1D3335] text-base leading-relaxed text-center max-w-2xl mx-auto mb-8">
            Mothra is an open-source annotation and Optical Music Recognition
            (OMR) platform developed at the DDMAL Lab, McGill University. It is
            designed to support the full analysis pipeline for medieval music
            manuscripts — from image annotation and training data creation
            through to automated layout detection — using modern YOLO-based
            object detection.
          </p>
          <p className="text-center text-sm italic text-[#1D3335] mb-8">
            jump to:{" "}
            <a
              href="#research-context"
              onClick={(e) => {
                e.preventDefault();
                scrollToCenter("research-context");
              }}
              className="underline hover:opacity-70 transition-opacity mx-2"
            >
              research context
            </a>
            <a
              href="#project-history"
              onClick={(e) => {
                e.preventDefault();
                scrollToCenter("project-history");
              }}
              className="underline hover:opacity-70 transition-opacity mx-2"
            >
              project history
            </a>
            <a
              href="#design-philosophy"
              onClick={(e) => {
                e.preventDefault();
                scrollToCenter("design-philosophy");
              }}
              className="underline hover:opacity-70 transition-opacity mx-2"
            >
              design philosophy
            </a>
          </p>
          <hr className="border-white/30" />
        </div>
      </section>

      {/* Research Context */}
      <section
        id="research-context"
        className="scroll-fade fade-target px-6 py-10"
      >
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl sm:text-3xl font-bold italic text-white text-right mb-6">
            research context
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 mb-8">
            <div className="space-y-4">
              <p className="text-[#1D3335] text-sm leading-relaxed">
                Medieval music manuscripts present a distinct set of challenges
                for automated recognition: degraded parchment, overlapping
                notation systems, significant variation in scribal hands, and
                layouts that resist rule-based segmentation. Mothra addresses
                these challenges by applying object detection techniques
                developed for complex document layouts to the specific domain of
                medieval musical sources.
              </p>
              <p className="text-[#1D3335] text-sm leading-relaxed">
                The project is positioned as an experimental alternative to the
                existing DDMAL pipeline, Rodan, which relies on multi-stage
                pixel-level segmentation. Where Rodan offers fine-grained
                control at each processing stage, Mothra investigates whether
                end-to-end deep learning can achieve comparable or superior
                results with lower annotation overhead and greater robustness to
                manuscript degradation.
              </p>
            </div>
            <div className="bg-[#1E6B70] rounded-lg h-48 sm:h-auto" />
          </div>
          <hr className="border-white/30" />
        </div>
      </section>

      {/* Project History */}
      <section
        id="project-history"
        className="scroll-fade fade-target px-6 py-10"
      >
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl sm:text-3xl font-bold italic text-white text-center mb-6">
            project history
          </h2>
          <div className="max-w-xl mx-auto text-center space-y-4 mb-8">
            <p className="text-[#1D3335] text-sm leading-relaxed">
              Mothra was initiated at the{" "}
              <a
                href="https://ddmal.ca"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:opacity-70 transition-opacity"
              >
                DDMAL Lab (Distributed Digital Music Archives and Libraries)
              </a>{" "}
              at McGill University's Schulich School of Music. It builds on the
              lab's existing expertise in computational musicology and the Rodan
              OMR ecosystem, while exploring a methodologically distinct
              approach grounded in contemporary object detection research.
            </p>
            <p className="text-sm italic text-[#1D3335]">
              visit the lab website:{" "}
              <a
                href="https://ddmal.ca"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:opacity-70 transition-opacity"
              >
                DDMAL
              </a>
            </p>
          </div>
          <hr className="border-white/30" />
        </div>
      </section>

      {/* Design Philosophy */}
      <section
        id="design-philosophy"
        className="scroll-fade fade-target px-6 py-10 pb-16"
      >
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl sm:text-3xl font-bold italic text-white mb-8">
            design philosophy
          </h2>
          <div className="space-y-8">
            <div>
              <h3 className="text-lg italic text-white mb-3 pl-4">
                annotation-first
              </h3>
              <p className="text-[#1D3335] text-sm leading-relaxed pl-8">
                The annotation tool is browser-based and requires no local
                installation. It is designed to minimize the time between
                obtaining a manuscript image and producing usable training data,
                with keyboard-driven workflows and three semantic annotation
                classes: text regions, music systems, and staff lines.
              </p>
            </div>
            <div>
              <h3 className="text-lg italic text-white mb-3 pl-4">
                manuscript-aware methodology
              </h3>
              <p className="text-[#1D3335] text-sm leading-relaxed pl-8">
                Training and evaluation are organized around the structure of
                manuscript corpora rather than individual pages. Data splits are
                performed at the manuscript level to prevent information leakage
                between training and evaluation sets, and model performance is
                assessed against held-out manuscripts rather than held-out
                pages.
              </p>
            </div>
            <div>
              <h3 className="text-lg italic text-white mb-3 pl-4">
                collaborative by design
              </h3>
              <p className="text-[#1D3335] text-sm leading-relaxed pl-8">
                The annotation tool is accessible to non-technical contributors
                via a shared URL with no setup required, making it suitable for
                distributed research teams. Annotations are saved client-side
                and exportable in both JSON and YOLO formats.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
