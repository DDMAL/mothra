# Embedded Neon update — issue #215

Date: 2026-11-05 (America/Toronto)

The embedded editor uses Neon develop `68e3555b` plus the reviewed cache
pending-save fix. Successful notation conversions now send the changed MEI to
the content API, preserve it across navigation and a fresh edit session, and
complete their successful PUT response before Mark Done sends corrected PATCH.
The final build and all six repository integration tests pass locally.

## Pinned version and changes

- Mothra base: `7ac370af52ec51da82b95bea633a0904daffc621`.
- Neon pin: `6b168f444e975c19a772981c4ceb3d06aae8022a` →
  `1b6e34ab99cfcdbfc444f5315467ebf3844e3bc3`.
- The new Neon commit has parent `68e3555b0a98f0f417629f8fbdca245511ee7846`.
  It preserves later merged changes, including the `l` connection shortcut.
  It is not a pin directly to the older source commit `2a46faaf`.
- Its only source change is the five-line `NeonCore.ts` patch from
  `2a46faaf956bec1f525e53b59c2bba85cfc00e25`. The original author, author date,
  commit explanation and coauthor are retained, with a `Cherry-picked-from`
  trailer recording the source.
- Bundled Verovio: `5.4.0-dev-ac22768-dirty` → `6.4.0-dev-ce49d38`.
- Remove Mothra's extra notation-dropdown click on iframe load. Neon initializes
  a new manifest from its MEI; replaying stale React `xmlContent` can otherwise
  convert a saved document back to its old notation.
- Keep Mothra's build command, Save/Mark Done implementation, fonts and
  insert-mode guard. The legacy save-helper docstring changes; its behavior does
  not. Control points, mapping and notation/subtype migration are outside #215.

## Pending-save correction

Changing notation calls `replaceMEI`, which marks the cache entry dirty. Changing
font then clears `lastPageLoaded`, and rendering reloads the cached MEI.
`loadPage` now passes `cached.dirty` to `loadData` rather than resetting it to the
default false. A render therefore preserves the need to save, and
`updateDatabase` includes the changed page in its PUT requests.

This source patch is byte-for-byte identical to the reviewed candidate patch on
`68e3555b`. There are no additional Neon source changes in the new commit.

## Final validation

| Check | Result |
| --- | --- |
| Full `landing-page` build: Neon, TypeScript and Vite | Passed with Node 22.16.0; existing large-chunk warning |
| Initial Square/Hufnagel selection and unedited page navigation | Passed, no writes |
| Square → Hufnagel through Save control, page navigation and fresh session | Passed; raw PUT and newly fetched MEI checked |
| Hufnagel → Square through Save control, page navigation and fresh session | Passed; raw PUT and newly fetched MEI checked |
| Cancel/confirm unsaved-navigation dialog | Passed for the existing dialog and no-write assertions |
| Mark Done after conversion | Passed; successful PUT response completes before corrected PATCH |
| Native Hufnagel connection edits, Mark Done and reopen | Passed; changed connection state and successful PUT-response order checked |

The repository suite contains six tests, with no skipped tests. Important
assertions from the separately reviewed candidate tests are now in
`landing-page/tests/neon-editor`; they require no personal-workspace paths or
files. Both specs observe actual worker completion before interactions, avoiding
selection races while the initial font/render is still loading. The conversion
cases also assert the MEI passed to Verovio, the raw saved XML, new session
identities, and GET responses matching that saved XML.

The tests use the actual React wrapper and built Neon iframe, with local API
responses and blank manuscript images supplied by Cypress. They exercise
successful saves with OK responses. They do not establish correctness for failed
saves, timeouts, Neon's local fallback, authentication, Postgres, the full app
router or Gen's production project. The unsaved-navigation test does not establish
full data consistency after discarding edits. No production data was accessed.
Commands and setup are in `landing-page/tests/neon-editor/README.md`.

## Built assets and reviewed candidate

The normal Mothra build uses Neon's development webpack mode and the pinned local
Verovio worker. Using Neon's `build:prod` instead would load a moving online
Verovio develop URL and change the tested dependency.

The reviewed candidate had 94 built Neon assets. In the final clean-cache build,
91 are byte-for-byte identical. The remaining three JavaScript bundles have the
same executable content after accounting for the displayed Neon version label;
their source maps differ only in source paths, with identical embedded source
contents. The old webpack cache retained a `7beb021e` display label. The final
editor correctly displays `1b6e34ab`. These metadata differences do not represent
additional product logic changes.

Neon's build output, Mothra `public/neon` and final `dist/neon` copies match. The
final editor SHA256 is
`f6b7d2523c08b9f15a839305b7be3e2dfda7a86b84612dd784429433cee1272a`.
The unchanged local Verovio bundle SHA256 is
`7689c9549b49a0f233bca23f3ea2b9e9951f2ad49a2d92b8865a824aace8f722`.
Machine-readable comparison results are in `NEON_215_ASSET_COMPARISON.json`.

## Delivery order

Both changes are local commits only. No push, pull request or deployment has been
performed. The new Neon commit must first become reachable from the configured
submodule remote, `https://github.com/DDMAL/Neon.git`. Until then, another clone or
CI cannot fetch this Mothra pin from that remote. Publish the Neon commit first
when authorized, then publish the Mothra change pointing to it. A live-backend
check remains separate from the local successful-save tests above.
