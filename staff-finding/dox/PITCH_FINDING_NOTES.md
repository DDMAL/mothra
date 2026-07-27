# Notes on Pitch Finding

What staff-finding gives us and what pitch finding would need.

These notes assume the target repertoire is square notation on a 4-line staff
(Gregorian chant / Hildegard).

---

## What we already have from staff-finding

The JSOMR JSON output from any of the runners provides, per fitted line:

- **`centerline`** — y-value at every integer x-column across the box x-range.
  This is a dense per-column position table, not just a bounding box.
- **`scale_unit`** — the median inter-staffline spacing in pixels, computed
  globally from the gap distribution.  This is the pitch-grid step size.
- **`stave_id`** and **`within_stave_index`** — which stave each line belongs to
  and its order within that stave (0 = bottom in most square-notation
  conventions, but confirm from the ADR / JSOMR spec).
- **`bounding_box`** — x-range where the fit is valid.

So the output already encodes a **per-column pitch grid**: for any note head at
position (x, y), you can query the five nearest staffline y-values at column x
and compute pitch by distance in units of `0.5 × scale_unit` (each semistep is
one half-space on the staff).

---

## What pitch finding would additionally need

### 1. Note-head (neume) detection and localisation

A note's pitch is its y-centre relative to the staff.  You need the y-centre
of each note head in image coordinates.  For square notation this is relatively
easy compared to modern notation: square noteheads have compact, regular shapes
and a YOLO detector trained on staffline boxes could be adapted.  Alternatively,
the existing `text_music_detector_fulldata.pt` model already distinguishes
text from music regions — that's a starting point.

Key subtlety: neumes are groups of notes written as a single connected shape
(podatus, clivis, scandicus, etc.).  Pitch finding has to parse *individual
note positions within a neume*, not just neume bounding boxes. A clivis is made of
two distinct square blobs: blob 1 is pitch 1, and blob 2 is pitch 2. They 
are connected, etc. Sampling central gravity might work. 

In experiments/dp_tracing, you might find some interesting ways forward from here on. 
The method used is terrible stafflines, but given how it works might be quite
good for identifying pitches in neumes (AKA, neume components, "nc").

### 2. Clef detection and clef-position mapping

Square notation uses C and F clefs (and occasionally Bb).  The clef sits on
one of the four staff lines and tells you the absolute pitch of that line.
Without the clef you only know *relative* pitch (interval from one note to the
next), not the *absolute* pitch name.

Clef detection is a classification problem on image crops: given a region at
the left edge of a stave, identify {C-clef, F-clef, Bb-clef, none} and the
staff line it sits on (line 1–4 from bottom).

Once you have:
  - clef type → pitch name of the reference line
  - clef staff-line index → which `within_stave_index` it sits on
  - note y-position → distance in half-spaces from that reference line

...you can assign a pitch letter (and octave register with a bit more work).
Proof of concept stage, we don't need to be picky about octave. That can 
wait a bit.

### 3. Accurate inter-line y-positions at the note's x-column

This is exactly what the interpolation pass (the current main TODO) provides.
If lines are missing from the staff, the pitch grid has holes and pitches that
fall in those holes will be mis-assigned.  **Completing the interpolation pass
is a prerequisite for robust pitch finding.**

The centerline output already gives per-column y-values for each detected line.
A pitch-grid builder would:
1. For each x-column, collect the y-values of all lines in a stave.
2. Fill any missing lines via `interpolate_staves.py` (once implemented).
3. Expose a function `pitch_at(stave_id, x, y_obs)` → half-space offset from
   the reference line.

### 4. Handling accidentals and custos

Square notation uses flat and natural signs as inline accidentals.  These sit
on a staff position and modify the pitch of the following note.  A pitch finder
would need to detect and localise them to output correct pitch names rather than
just scale degrees.

Don't spend too much time on accidentals early on, perfect accidental carryover
is a stretch goal, and not something I'm concerned about in the proof of concept
stage. 

Custos (the small guide symbol at the end of a line indicating the first pitch
of the next line) should probably be excluded from pitch assignment.

---

## Suggested architecture

```
Staff-finding output (JSOMR)
        ↓
Pitch grid builder
  - per stave, per x-column: sorted list of staffline y-values
  - fills missing lines (requires interpolation pass)
        ↓
Neume / note-head detector
  - outputs: (x_centre, y_centre, neume_type) per neume symbol
  - neume segmentation: split composite neumes into individual note y-positions
        ↓
Clef detector
  - input: left-edge crop of each stave
  - output: (clef_type, clef_line_index) per stave
        ↓
Pitch resolver
  - for each note (stave_id, x, y_obs):
      offset = nearest_staffline_index(x, y_obs) based on pitch grid
      pitch  = clef_reference_pitch + offset (in half-spaces)
  - outputs MIDI pitch or kern/mei pitch name
```

---

## Data we have that would help

- **`addtl-gt/`** — 84 annotated pages with YOLO staffline boxes.  Several
  manuscripts in this set (Einsiedeln, Montecassino, PennLJS418, Plimpton041,
  l'Arsenal) are square-notation chant books and could provide neume training
  data if bounding-box annotations for note heads exist or can be added.

- **`image-sets/gent/`** — the Ghent Antiphoner is a high-quality digitisation
  of a well-known manuscript; a potentially good "new" test manuscript. 

- **`e2e-OMR-resources`** GitHub repository — holds all of the corrected MEI 
for Einsiedeln, Salzinnes, and MS73. If you have questions about the 
structure of these directories or the files within, talk to Gen. 

- **Cantus Database spreadsheets** — Use a source that is published, 
with images, with music. MS73 will be pretty perfect for this: it has
Volpiano available through CantusDB (and so, through the .csv you
can download from the source page), as well as a set of corrected 
.mei files.

- **`models/text_music_detector_fulldata.pt`** — already separates text from
  music regions; may accelerate note-head localisation by restricting the search
  area. As a note, give this a padding zone of about 5-10 pixels, initially: some
  scribes can get very slanty/shunted over at they go, and don't align
  precisely over the syllable they are connected to.

---

## The one thing to get right first
The transition to establishing YOLO for staff-finding cannot efficiently
and consistently enough grab stafflines to a segmentation model will be 
the single largest blocker for this work. The staff finding script(s) work
well so long as the majority of the stafflines have bounding boxes. Finishing
the interpolation step with assist additionally with this. 

The pitch grid is only as good as the staff-finding output.  Dropped stafflines
produce ambiguous pitch assignments for notes that fall in the gap.  Completing
`interpolate_staves.py` and running it across the test set is the single most
valuable prerequisite for pitch finding.  Everything else (clef detection,
neume segmentation) can be built in parallel, but pitch accuracy will be gated
on interpolation quality.

**Additional thoughts**
I think it might be worth training a small model on how we decide what pitch
is on or off a line, in or out of a space. Doing that wholly heuristically
is giving nightmare fuel. If you'd like to give the pure heuristic approach, 
the code for the heuristic pitch finding job is inside of Rodan, and can
(and should) be consulted to see what can be utilised or adapted. 