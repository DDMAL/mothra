import { useEffect } from "react";

interface DocsWalkthroughProps {
  scrollTarget: string | null;
}

export default function DocsWalkthrough({ scrollTarget }: DocsWalkthroughProps) {
  useEffect(() => {
    if (!scrollTarget) return;
    document
      .getElementById(`wt-${scrollTarget}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [scrollTarget]);

  return (
    <div className="animate-fade-in px-10 pt-10 pb-16 text-[#1D3335]">
      <h1 className="text-3xl font-bold italic text-white mb-4">walkthrough</h1>
      <p className="leading-relaxed mb-10 max-w-2xl">
        This walkthrough follows the same order the project page's progress
        steps do: <em>models</em> and <em>annotations</em> come first (they
        can happen in either order, but you need at least one model before
        you can annotate), then <em>interactive classifier</em>,{" "}
        <em>encoding</em>, and <em>neon</em> proceed one page — or one
        batch — at a time, and <em>sending to Cantus Ultimus</em> closes out
        the project.
      </p>

      <section id="wt-models" className="mb-10">
        <h2 className="text-2xl font-bold italic text-white mb-4">models</h2>
        <p className="leading-relaxed mb-4">
          Mothra uses two unrelated kinds of model, both managed from the
          project's <em>models</em> tab:
        </p>
        <ul className="list-disc list-inside mb-4 space-y-2 pl-4">
          <li>
            <strong>YOLO detection models</strong> (<code>.pt</code> files) —
            used by the <em>annotations</em> step below to find text regions,
            music systems, and staff lines on a page image.
          </li>
          <li>
            <strong>Text-finding models</strong> — a segmentation model and a
            recognition/OCR model (both <code>.mlmodel</code> or{" "}
            <code>.safetensors</code>), plus an optional text-region mask (a{" "}
            <code>.json</code> file). These feed the Kraken-based
            handwritten-text-recognition pass that runs alongside YOLO
            detection.
          </li>
        </ul>
        <p className="leading-relaxed mb-4">
          You don't need to upload anything to get started: choosing the{" "}
          <strong>medieval manuscripts</strong> preset uses a pretrained
          text/music detector and stave detector that ship with Mothra and
          work fully offline. Pick <strong>custom</strong> instead if you
          want to bring your own YOLO checkpoint or text-finding models —
          upload them via "+ new model", either by dragging a file in or
          selecting one from your computer.
        </p>
        <p className="leading-relaxed">
          As with images, clicking a model gives you the option to "use" it —
          this is what determines which model actually runs when you get to
          the annotations step.
        </p>
      </section>

      <section id="wt-annotations" className="mb-10">
        <h2 className="text-2xl font-bold italic text-white mb-4">
          annotations
        </h2>
        <p className="leading-relaxed mb-4">
          This is the project's step 0 — the "annotate" button on the project
          page. It runs your selected YOLO model over every image you've
          marked as "used," drawing bounding boxes for text regions, music
          systems, and staff lines, and (if you've selected text-finding
          models too) running handwritten text recognition over the text
          regions at the same time.
        </p>
        <p className="leading-relaxed mb-4">
          If some of your used images already have annotations from a
          previous run — say, you added a new image to a project you'd
          already annotated — only the new, not-yet-annotated images are
          processed. Already-annotated images are left alone rather than
          being silently redone.
        </p>
        <p className="leading-relaxed mb-4">
          If your images are grouped under a Cantus source with folio numbers
          (set up during batch image upload), annotating runs as a single
          batch job across the whole source instead of one image at a time —
          this is the same detection + text-recognition pass, just processed
          together so the text-finding stage can take the whole source's
          layout into account.
        </p>
        <p className="leading-relaxed">
          Once annotation finishes, open the <em>annotations</em> tab on any
          image to see the detected boxes drawn directly over the source
          image. From there you can download the raw YOLO output as{" "}
          <code>.txt</code> or as JSON for a single image, or — from the
          project page, if your images belong to a Cantus source — export a
          zip of every image's annotations and text-alignment data for that
          whole source at once.
        </p>
      </section>

      <section id="wt-interactive-classifier" className="mb-10">
        <h2 className="text-2xl font-bold italic text-white mb-4">
          interactive classifier
        </h2>
        <p className="leading-relaxed mb-4">
          The interactive classifier (IC) is where individual music glyphs —
          neumes, clefs, custodes, and so on — get classified. It opens as an
          embedded tool, one page at a time, starting from the first used
          image that hasn't been through IC yet.
        </p>
        <p className="leading-relaxed mb-4">
          Before starting a session, you can optionally seed the classifier
          with training data — pick one of the built-in presets (like
          Hufnagel or Square notation), upload your own GameraXML training
          set, or both — and choose a vocabulary to control which class names
          are available. Press "start session" once you're ready.
        </p>
        <p className="leading-relaxed mb-4">
          Inside the session, glyphs are grouped by how confidently they were
          classified. Correct any mistakes by dragging a glyph to the right
          class, adjust <em>k</em> and hit "reclassify" to have the
          classifier reconsider its groupings with your corrections folded
          in, and repeat until you're happy with the page.
        </p>
        <div className="w-full h-64 bg-[#1E6B70] rounded-lg mb-4" />
        <p className="leading-relaxed mb-4">
          When a page is ready, press <strong>"queue page"</strong> — this
          finalizes the session into a GameraXML file behind the scenes and
          adds the page to a queue, then automatically advances you to the
          next un-queued page so you can keep going without leaving the
          classifier. The clef shape and clef line selectors at the top apply
          to every page you queue.
        </p>
        <p className="leading-relaxed">
          Once you've queued however many pages you want to encode together —
          anywhere from one page to the whole project — press{" "}
          <strong>"encode batch (N)"</strong> to send all of them into
          encoding at once. There's no requirement to queue more than one
          page first: queuing a single page and encoding a "batch" of one
          works the same way it always did.
        </p>
      </section>

      <section id="wt-encoding" className="mb-10">
        <h2 className="text-2xl font-bold italic text-white mb-4">
          encoding
        </h2>
        <p className="leading-relaxed mb-4">
          Encoding turns each queued page's GameraXML output into an MEI
          file — pitch-less music notation with the neume classifications
          from IC, staff positions estimated from either your YOLO stave
          annotations or the glyph layout itself, and any text-finding
          results aligned underneath.
        </p>
        <p className="leading-relaxed mb-4">
          For a batch, this page shows which item is currently encoding
          ("encoding 2 of 5," and so on) along with a live log of each
          page's progress. If one page in a batch fails to encode — a
          malformed XML file, for instance — the rest of the batch keeps
          going; you'll get a summary at the end showing how many pages
          succeeded and which ones, if any, failed.
        </p>
        <p className="leading-relaxed">
          Every successfully encoded page becomes an MEI file on the project,
          ready for the correction step below. You don't need to wait for a
          whole batch to finish before starting to correct the pages that are
          already done.
        </p>
      </section>

      <section id="wt-neon" className="mb-10">
        <h2 className="text-2xl font-bold italic text-white mb-4">neon</h2>
        <p className="leading-relaxed mb-4">
          This step opens an embedded copy of{" "}
          <a
            href="https://github.com/DDMAL/Neon"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:opacity-70 transition-opacity"
          >
            Neon
          </a>
          , DDMAL's MEI editor, for manually correcting whatever the
          automated pipeline got wrong — misplaced neumes, incorrect staff
          positions, missing text alignment, and so on.
        </p>
        <p className="leading-relaxed mb-4">
          You step through your project's MEI files one at a time. Once
          you're satisfied with a file, mark it corrected before moving to
          the next one — this is what the rest of the app uses to tell
          "encoded but not yet reviewed" apart from "reviewed and ready to
          send."
        </p>
        <p className="leading-relaxed">
          Press "finish" once every file you care about has been reviewed to
          move on to sending.
        </p>
      </section>

      <section id="wt-sending-mei-files-to-cantus-ultimus" className="mb-10">
        <h2 className="text-2xl font-bold italic text-white mb-4">
          sending mei files to cantus ultimus
        </h2>
        <p className="leading-relaxed mb-4">
          This step marks the project as finished in Mothra and points you to{" "}
          <a
            href="https://cantus.simssa.ca/"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:opacity-70 transition-opacity"
          >
            Cantus Ultimus
          </a>
          , where corrected MEI files ultimately become searchable by music
          and text.
        </p>
        <p className="leading-relaxed">
          Right now this is a hand-off, not an automated upload — Mothra
          doesn't push files into Cantus Ultimus for you yet. Use the
          exporting options below to get your corrected MEI files off of
          Mothra, and get in touch with whoever manages ingestion into Cantus
          Ultimus for your source to have them added. Direct, in-app sending
          is planned but not built yet.
        </p>
      </section>

      <section id="wt-saving-and-exporting-data" className="mb-10">
        <h2 className="text-2xl font-bold italic text-white mb-4">
          saving and exporting data
        </h2>
        <p className="leading-relaxed mb-4">
          Nothing you do in Mothra requires you to export anything to keep
          it — everything is saved to your project as you go. Exporting is
          purely for getting copies of your data out. A few ways to do that:
        </p>
        <ul className="list-disc list-inside mb-4 space-y-2 pl-4">
          <li>
            <strong>A single MEI file or its Neon manifest</strong> — download
            either right after encoding a page, from the encoding-complete
            screen.
          </li>
          <li>
            <strong>A single image's annotations</strong> — from the{" "}
            <em>annotations</em> tab, as raw YOLO <code>.txt</code> or as
            JSON.
          </li>
          <li>
            <strong>A whole Cantus source's annotations and text
            alignment</strong> — one zip covering every image tagged with
            that source, generated fresh from whatever's currently saved (so
            it always reflects your latest corrections, not just the results
            of the last batch run).
          </li>
          <li>
            <strong>The whole project</strong> — a zip of every MEI file plus
            its manifest, and a separate zip of the project's activity logs,
            both available from the project page.
          </li>
        </ul>
        <p className="leading-relaxed">
          Deleting a project doesn't destroy your data immediately — it moves
          to a trash folder in "my projects," where you can restore it later
          if needed, or leave it to be permanently removed after a while.
        </p>
      </section>

      <section id="wt-user-accounts" className="mb-16">
        <h2 className="text-2xl font-bold italic text-white mb-4">
          user accounts
        </h2>
        <p className="leading-relaxed mb-4">
          Registering needs a username, password, email address, and your
          first and last name. Once logged in, the dropdown under your name
          in the navbar gives you access to <em>my account</em>, where you
          can update your profile, change your password, or delete your
          account entirely.
        </p>
        <p className="leading-relaxed">
          One thing worth knowing: logging in keeps you signed in for 72
          hours, and there's currently no way to refresh that without
          logging in again. If you get signed out mid-workflow, don't worry —
          your project's progress is saved step by step, so logging back in
          and reopening the project picks up right where you left off.
        </p>
      </section>
    </div>
  );
}
