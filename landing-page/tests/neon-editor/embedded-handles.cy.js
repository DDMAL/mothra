// Adapted from Neon #1410 cypress/e2e/edit-handles.cy.ts and support/neon-mei.ts.
// The same samples, measurements and gestures run in Mothra's production iframe.
import { observeNeonRenderer, waitForNeonRenderer } from './helpers/renderer';

function element(selector) {
  return cy.get('iframe').should(($frames) => {
    const frame = $frames[0];
    expect(frame.contentDocument?.URL).to.equal(frame.src);
    expect(frame.contentDocument?.querySelector(selector), selector).to.exist;
  }).then(($frames) => cy.wrap($frames[0].contentDocument).find(selector));
}

function openEditor(notation = 'square', scale = 1) {
  const session = `handles-${Date.now()}-${Cypress._.random(1000000)}`;
  const path = '/api/projects/215/mei/handles/content?token=local-test';
  const image = `${Cypress.config('baseUrl')}/__test__/handles-page.svg`;
  const width = Math.round(4872 * scale), height = Math.round(6496 * scale);
  cy.task('readNeonFixture', notation).then((original) => {
    let xml = notation === 'square'
      ? original.replace('notationtype="neume"', 'notationtype="neume.square"') : original;
    if (scale !== 1) xml = xml.replace(/\b(ulx|uly|lrx|lry)="(-?\d+(?:\.\d+)?)"/g,
      (_match, attr, value) => `${attr}="${Math.round(Number(value) * scale)}"`);
    cy.intercept('GET', '/__test__/files', { body: { files: [
      { id: 'handles', name: `${notation}-handles.mei`, xmlContent: xml, corrected: false },
    ] } });
    cy.intercept('POST', '/api/projects/215/mei/handles/edit-session', {
      body: { session_id: session, manifest_id: session },
    }).as('handleSession');
    cy.intercept('GET', `/neon/samples/manifests/${session}.jsonld`, { body: {
      '@context': ['http://www.w3.org/ns/anno.jsonld', {
        schema: 'http://schema.org/', title: 'schema:name', timestamp: 'schema:dateModified',
        image: { '@id': 'schema:image', '@type': '@id' },
        mei_annotations: { '@id': 'Annotation', '@type': '@id', '@container': '@list' },
      }],
      '@id': `urn:uuid:${session}`, title: 'Embedded handles', image,
      mei_annotations: [{ id: `urn:uuid:annotation-${session}`, type: 'Annotation',
        body: `${Cypress.config('baseUrl')}${path}`, target: image }],
    } });
    cy.intercept('GET', '/api/projects/215/mei/handles/content*', {
      headers: { 'content-type': 'application/xml' }, body: xml,
    });
    cy.intercept('GET', '/__test__/handles-page.svg', {
      headers: { 'content-type': 'image/svg+xml' },
      body: `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="100%" fill="white"/></svg>`,
    });
  });
  observeNeonRenderer();
  cy.visit('/tests/neon-editor/index.html');
  cy.wait('@handleSession');
  element('svg.active-page .nc');
  waitForNeonRenderer();
  element('#bgimg').should('have.attr', 'width', String(width));
}

function selectStaff() {
  waitForNeonRenderer();
  element('#selByStaff').click({ force: true }).should('have.class', 'is-active');
  element('.staff').first().then(($staff) => {
    const boxes = [...$staff[0].querySelectorAll('path')].map((p) => p.getBoundingClientRect());
    const x = (Math.min(...boxes.map((b) => b.left)) + Math.max(...boxes.map((b) => b.right))) / 2;
    const y = (Math.min(...boxes.map((b) => b.top)) + Math.max(...boxes.map((b) => b.bottom))) / 2;
    element('#svg_group').then(($g) => {
      const box = $g[0].getBoundingClientRect();
      cy.wrap($g).click(x - box.left, y - box.top, { force: true });
    });
  });
  element('.staff').first().should('have.class', 'selected');
  element('#rotateLeft');
}

function zoomTo(percent) {
  element('#zoomSlider').then(($slider) => {
    const slider = $slider[0];
    slider.value = String(percent);
    slider.dispatchEvent(new slider.ownerDocument.defaultView.MouseEvent('mouseup', { bubbles: true }));
  });
}

function handleSize(size) {
  element('#handleSizeSlider').invoke('val', size).trigger('input');
  element('#handleSizeOutput').should('have.value', String(size));
}

function metrics(doc) {
  const circle = doc.querySelector('#p-Left').getBoundingClientRect();
  const diamond = doc.querySelector('#rotateLeft').getBoundingClientRect();
  return { circle: circle.width, diamondW: diamond.width, diamondH: diamond.height,
    offsetX: diamond.x + diamond.width / 2 - circle.x - circle.width / 2,
    offsetY: diamond.y + diamond.height / 2 - circle.y - circle.height / 2 };
}

function measure(size, label) {
  element('#p-Left').should(($c) => {
    const m = metrics($c[0].ownerDocument);
    for (const [key, expected] of Object.entries({ circle: 2 * size, diamondW: 2.4 * size,
      diamondH: 2.8 * size, offsetX: -4 * size, offsetY: -2 * size })) {
      expect(m[key], `${label} ${key} CSS px`).to.be.closeTo(expected, 0.6);
    }
  }).then(($c) => cy.task('recordHandleResult', { case: label, size,
    cssPixels: metrics($c[0].ownerDocument) }));
}

function readMei() {
  element('#getmei').then(($link) => {
    const link = $link[0], win = link.ownerDocument.defaultView;
    link.setAttribute('href', '#');
    link.addEventListener('click', (e) => e.preventDefault(), { once: true });
    link.dispatchEvent(new win.MouseEvent('click', { bubbles: true, cancelable: true }));
  });
  return element('#getmei').should(($link) => expect($link.attr('href')).to.match(/^data:/))
    .then(($link) => new DOMParser().parseFromString(atob($link.attr('href').split(',')[1]), 'text/xml'));
}

function content(doc) {
  const music = doc.querySelector('music').cloneNode(true);
  music.querySelectorAll('pb').forEach((p) => p.removeAttribute('xml:id'));
  return new XMLSerializer().serializeToString(music);
}

function sameContent(actual, expected) {
  const a = content(actual), b = content(expected);
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  expect(a.length, 'MEI content length').to.equal(b.length);
  expect(a.slice(Math.max(0, i - 60), i + 60), `MEI content at ${i}`)
    .to.equal(b.slice(Math.max(0, i - 60), i + 60));
}

function zone(doc, id) {
  const staff = [...doc.querySelectorAll('*')].find((e) => e.getAttribute('xml:id') === id);
  // Parsed XML nodes are not attached to the browser DOM; use identity rather
  // than Cypress's DOM-aware "exist" assertion (as in upstream staffZone).
  expect(staff, `staff ${id} in exported MEI`).not.to.equal(undefined);
  const zoneId = staff.getAttribute('facs').replace('#', '');
  const z = [...doc.querySelectorAll('zone')].find((e) => e.getAttribute('xml:id') === zoneId);
  return Object.fromEntries(['ulx', 'uly', 'lrx', 'lry'].map((a) => [a, Number(z.getAttribute(a))]));
}

function edits() {
  return cy.get('iframe').its('0.contentWindow').then((win) =>
    win.__neonTestMessages.filter((m) => m.action === 'edit'));
}

function dragBy(dx, dy) {
  element('.staff.selected').then(($staff) => {
    const staff = $staff[0], win = staff.ownerDocument.defaultView;
    const k = staff.getScreenCTM().a;
    const g = staff.getBoundingClientRect(), line = staff.querySelector('path').getBoundingClientRect();
    const x = line.left + line.width / 2 - g.left, y = line.top + line.height / 2 - g.top;
    cy.wrap($staff).trigger('mousedown', x, y, { force: true, which: 1, view: win })
      .trigger('mousemove', x + dx * k, y + dy * k, { force: true, view: win })
      .trigger('mouseup', { force: true, view: win });
  });
  // d3-drag suppresses the next click for one event-loop task after mouseup.
  cy.wait(100);
}

describe('Mothra embedded Neon #1410', () => {
  for (const notation of ['square', 'hufnagel']) {
    it(`${notation}: fixed CSS size through Zoom; hidden bbox slider; MEI unchanged`, () => {
      openEditor(notation);
      readMei().then((before) => {
        selectStaff();
        handleSize(6);
        for (const zoom of [100, 25, 400, 100]) {
          zoomTo(zoom);
          measure(6, `${notation} zoom ${zoom}%`);
        }
        element('#displayBBox').should('not.be.checked').click({ force: true }).click({ force: true });
        element('#edit-handle-size-slider').scrollIntoView().should('be.visible');
        handleSize(10);
        measure(10, `${notation} bbox hidden size 10`);
        selectStaff();
        measure(10, `${notation} reselected`);
        readMei().then((after) => sameContent(after, before));
        edits().then((e) => expect(e, 'no editor action for UI changes').to.have.length(0));
      });
    });
  }

  for (const [scale, zoom, dx] of [[1, 100, -544], [1, 25, -544], [0.1, 25, 20]]) {
    it(`legal content drag with off-image handle: scale ${scale}, zoom ${zoom}`, () => {
      openEditor('square', scale);
      zoomTo(zoom);
      selectStaff();
      if (scale < 1) handleSize(20);
      element('.staff.selected').then(($s) => {
        const id = $s[0].id, k = $s[0].getScreenCTM().a;
        const tolerance = Math.max(3, 1.5 / k);
        element('#rotateLeft').then(($d) => {
          expect($d[0].getBBox().x + dx, 'handle lies beyond image').to.be.lessThan(0);
        });
        readMei().then((before) => {
          const start = zone(before, id);
          dragBy(dx, 0);
          waitForNeonRenderer();
          element('.staff path').first().then(($p) => {
            expect($p[0].ownerDocument.body.textContent).not.to.contain('Drag action failed');
          });
          readMei().then((after) => {
            const end = zone(after, id);
            expect(end.ulx - start.ulx).to.be.closeTo(dx, tolerance);
            expect(end.lrx - start.lrx).to.be.closeTo(dx, tolerance);
            expect(end.uly).to.be.closeTo(start.uly, tolerance);
            cy.task('recordHandleResult', { case: 'legal drag', scale, zoom, requestedDx: dx,
              tolerance, start, end });
          });
          edits().then((e) => expect(e, 'legal edit sent').to.have.length(1));
        });
      });
    });
  }

  for (const kind of ['glyphs', 'staff lines']) {
    it(`rejects actual ${kind} out of image and restores MEI`, () => {
      openEditor();
      selectStaff();
      readMei().then((before) => {
        element('.staff.selected').then(($s) => {
          const staff = $s[0], doc = staff.ownerDocument;
          const d = staff.querySelector('path').getAttribute('d');
          if (kind === 'staff lines') dragBy(-700, 0);
          else {
            const imageTop = doc.querySelector('#bgimg').getBoundingClientRect().top;
            const top = (sel) => Math.min(...[...staff.querySelectorAll(sel)]
              .map((e) => e.getBoundingClientRect()).filter((r) => r.width || r.height).map((r) => r.top));
            const glyphTop = top('use'), lineTop = top('path');
            expect(glyphTop).to.be.lessThan(lineTop - 4);
            dragBy(0, -(glyphTop - imageTop + (lineTop - glyphTop) / 2) / staff.getScreenCTM().a);
          }
          element('#notification-content').should('contain.text', 'Drag action failed');
          element('.staff path').first().should('have.attr', 'd', d);
          element('.staff').first().should('not.have.attr', 'transform');
          readMei().then((after) => sameContent(after, before));
          edits().then((e) => expect(e, 'invalid edit refused before toolkit').to.have.length(0));
          cy.task('recordHandleResult', { case: 'rejected drag', kind, meiRestored: true });
        });
      });
    });
  }
});
