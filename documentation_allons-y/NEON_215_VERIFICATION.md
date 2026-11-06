# Embedded Neon update — issue #215

Date: 2026-11-05 (America/Toronto)

This branch updates the embedded editor and verifies the ordinary Hufnagel
connection-editing path. It also exposes an upstream save defect when a user
changes notation type. **Keep this branch for review; resolve that defect before
deploying this combination.** No deployment was performed.

## Change

- Mothra base: `7ac370af52ec51da82b95bea633a0904daffc621`.
- Neon pin: `6b168f444e975c19a772981c4ceb3d06aae8022a` →
  `68e3555b0a98f0f417629f8fbdca245511ee7846` (merged upstream develop).
- Bundled Verovio: `5.4.0-dev-ac22768-dirty` → `6.4.0-dev-ce49d38`.
- Remove Mothra's extra notation-dropdown click on iframe load. In this Neon
  version the dropdown converts the MEI, so replaying stale `xmlContent` props
  can undo a user's saved change of notation. Neon already initializes new
  manifests from their MEI and maintains settings per manifest.
- Keep the normal build command, font assets, save/Mark Done handling, and
  insert-mode guard. No patches to Neon itself, mapping CSV, or notation migration.
- Update the legacy save-helper docstring; its behavior is unchanged.

The normal Mothra build uses Neon's development webpack mode and **the pinned
local Verovio bundle**. Switching to Neon's `build:prod` would instead load the
moving Verovio develop URL and would change the tested dependency. The source,
copied public asset, and final `dist` asset had identical SHA256:
`7689c9549b49a0f233bca23f3ea2b9e9951f2ad49a2d92b8865a824aace8f722`.

## Validation

| Check | Result |
| --- | --- |
| Full `landing-page` build, including Neon, TypeScript and Vite | Passed (Node 22.16.0); existing large-chunk warning |
| ESLint on modified `NeonBatchEditor.tsx` | Passed |
| Full-repository ESLint | Existing backlog: 51 errors, 8 warnings; unrelated files were not changed |
| MEI encoder and Neon manifest Python tests | 41 passed (Python 3.14) |
| Neon's `toggle-shortcut.cy.ts` and `drag/hufnagel-ligature.cy.ts` | 17 passed |
| Real Mothra wrapper + built Neon: initial Square/Hufnagel, switch pages without edits | Passed |
| Real wrapper: cancel/confirm leaving an unsaved notation change | Passed |
| Real wrapper: native Hufnagel connection editing, Mark Done and reload | Passed; raw PUT and subsequent GET checked, not only the Saved toast |
| Real wrapper: save after changing notation in either direction | Failed: no MEI PUT after font-switch render |
| Real wrapper: Mark Done after changing notation | Failed: no MEI PUT; corrected PATCH alone is insufficient |

Browser tests run the actual React component and bundled iframe, with local
session/content APIs supplied by Cypress. The correction callback updates the
file array as AppRouter does. They do not exercise a live database, authentication,
the complete application router, or Gen's production project. No production data
was read or written for these tests. Test setup and commands are in
`landing-page/tests/neon-editor/README.md`.

The initial Python attempt used the system Python 3.9, which rejected the
existing `zip(..., strict=True)` calls; rerunning under Python 3.14 passed all 41.
An early browser setup lacked the backend's `/Neon-gh` static alias; the dedicated
test server now mirrors that alias. Those environment failures are distinct from
the three reproducible save failures above.

## Deployment blocker: notation conversion loses its pending-save state

Reproduce in the integration suite by changing Square to Hufnagel (or the reverse),
waiting for the replacement SVG, then saving. The notation and glyphs change,
but no PUT is issued. Mark Done can subsequently mark the file corrected even
though that conversion did not reach the content API.

The source trace in Neon `68e3555b` is:

1. `NeonView.setNotationType` converts the MEI via `replaceMEI` (dirty=true).
2. `NeonCore.setNotationFont` clears `lastPageLoaded` when the font changes.
3. The ensuing render enters `NeonCore.loadPage`'s cached-page branch. It calls
   `loadData(pageURI, cached.mei)` without the cached dirty flag, resetting it to
   the default false.
4. `updateDatabase` saves only dirty entries, so the changed MEI is skipped.

A focused upstream correction should preserve the cache entry's dirty state
when reloading it for a font change, and test the actual PUT after rendering.
That Neon implementation is outside this Mothra-only branch. The failing tests
remain active so a later pin can be checked against the same reproduction.

## Other issues that this update does not resolve

Gen's actual Neon #1401 attachment was tested with this exact Verovio bundle:
all six connected pairs still share their zone after disconnecting. Dragging
either member moves both. Deleting either removes the shared zone while the
surviving nc still references it (12 deletion cases). Sending a chain of two
10-unit drags to a connected pair moves its shared zone by 20. These are toolkit
MEI/SVG reproductions, not a claim that Neon #1407 has the same cause.

Mothra #310 remains separate: current controlled mapping examples pass, but Gen's
specific 11/5 pre-Neon output was not attached to that issue and has not been
obtained. Updating Neon does not re-encode existing MEI or prove all mappings
correct. The planned standard-neume/subtype migration is also not included.
