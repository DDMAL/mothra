import { observeNeonRenderer, waitForNeonRenderer } from './helpers/renderer';

const projectPath = '/api/projects/215/mei';
const pagePath = '/tests/neon-editor/index.html';
const fileId = 'native-hufnagel';
const fileName = 'Mothra-Hufnagel-clivis.mei';
const firstNc = 'nc-g1';
const secondNc = 'nc-g1-1';
const connectorGlyph = /E9B[456789ABC]/;

function editorElement(selector) {
  return cy.get('iframe[title^="Neon editor - "]')
    .should(($frames) => {
      const frame = $frames[0];
      expect(frame.contentDocument?.URL, 'current iframe navigation loaded').to.equal(frame.src);
      expect(frame.contentDocument?.querySelector(selector), `${selector} in loaded iframe`).to.exist;
    })
    .then(($frames) => cy.wrap($frames[0].contentDocument).find(selector));
}

function expectConnected(connected) {
  editorElement(`svg.active-page #${firstNc} use, svg.active-page #${secondNc} use`)
    .should(($uses) => {
      const hrefs = $uses.toArray().map((use) =>
        use.getAttribute('xlink:href') || use.getAttribute('href') || '');
      expect(hrefs.some((href) => connectorGlyph.test(href)),
        `Hufnagel connector rendered (${hrefs.join(', ')})`).to.equal(connected);
    });
}

function expectSaved(saved) {
  editorElement('#file-saved').should('have.attr', 'alt',
    saved ? 'Your work is saved' : 'You have unsaved work');
}

function selectPair() {
  waitForNeonRenderer();
  editorElement(`#${firstNc} use`).first().click({ force: true });
  editorElement(`#${firstNc}`).should('have.class', 'selected');
  editorElement(`#${secondNc}`).then(($second) => {
    // Clicking either member of an already-connected pair selects both.
    if (!$second.hasClass('selected')) {
      editorElement(`#${secondNc} use`).first().click({
        force: true, metaKey: true, ctrlKey: true,
      });
    }
  });
  editorElement(`#${firstNc}`).should('have.class', 'selected');
  editorElement(`#${secondNc}`).should('have.class', 'selected');
}

function toggleConnection() {
  selectPair();
  cy.get('iframe').its('0.contentWindow').then((win) => {
    win.document.body.dispatchEvent(new win.KeyboardEvent('keydown', {
      key: 'l', code: 'KeyL', bubbles: true,
    }));
  });
}

function assertMei(xml, connected) {
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  expect(doc.querySelector('parsererror'), 'valid MEI').to.equal(null);
  expect(doc.querySelector('staffDef')?.getAttribute('notationtype'))
    .to.equal('neume.hufnagel');
  const components = [...doc.querySelectorAll('nc')];
  expect(components, 'both encoded components survive').to.have.length(2);
  for (const id of [firstNc, secondNc]) {
    expect(components.some((nc) => nc.getAttribute('xml:id') === id), id).to.equal(true);
  }
  const second = components.find((nc) => nc.getAttribute('xml:id') === secondNc);
  expect(second.getAttribute('con') === 'e', 'MEI connection state').to.equal(connected);
}

// Export through Neon's real control without downloading or navigating.
// Call after expectConnected(), which waits for the asynchronous edit/render.
function expectCurrentMei(connected) {
  editorElement('#getmei').then(($link) => {
    const link = $link[0];
    const win = link.ownerDocument.defaultView;
    link.setAttribute('href', '#');
    link.addEventListener('click', (event) => event.preventDefault(), { once: true });
    link.dispatchEvent(new win.MouseEvent('click', { bubbles: true, cancelable: true }));
  });
  editorElement('#getmei').should(($link) => {
    expect(($link.attr('href') || '').startsWith('data:'), 'MEI export ready').to.equal(true);
  }).then(($link) => {
    const encoded = $link.attr('href').split(',')[1];
    assertMei(atob(encoded), connected);
  });
}

describe('Mothra-generated Hufnagel connection editing in embedded Neon', () => {
  it('renders, toggles connections, and persists a changed connection state through Mark Done and reopen', () => {
    const server = {
      content: '', original: '', corrected: false, events: [], loads: [],
      saveResponseCompleted: false,
    };
    const run = `${Date.now()}-${Cypress._.random(1000000)}`;
    const manifests = new Map();
    const image = `${Cypress.config('baseUrl')}/__test__/native-page.svg`;
    let sessionNumber = 0;

    cy.task('readNeonFixture', 'mothraHufnagelClivis').then((xml) => {
      assertMei(xml, true);
      server.content = xml;
      server.original = xml;
    });
    cy.intercept('GET', '/__test__/files', (req) => req.reply({ body: {
      files: [{
        id: fileId, name: fileName,
        xmlContent: server.original, corrected: server.corrected,
      }],
      initialFileId: fileId,
    } }));
    cy.intercept('GET', '/__test__/native-page.svg', {
      headers: { 'content-type': 'image/svg+xml' },
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000"><rect width="100%" height="100%" fill="white"/></svg>',
    });
    cy.intercept('POST', `${projectPath}/${fileId}/edit-session`, (req) => {
      const session = `native-${run}-${++sessionNumber}`;
      const manifestId = `manifest-${session}`;
      manifests.set(session, {
        '@context': [
          'http://www.w3.org/ns/anno.jsonld',
          {
            schema: 'http://schema.org/', title: 'schema:name',
            timestamp: 'schema:dateModified',
            image: { '@id': 'schema:image', '@type': '@id' },
            mei_annotations: {
              '@id': 'Annotation', '@type': '@id', '@container': '@list',
            },
          },
        ],
        '@id': `urn:uuid:${manifestId}`,
        title: fileName,
        image,
        mei_annotations: [{
          id: `urn:uuid:annotation-${session}`,
          type: 'Annotation',
          body: `${Cypress.config('baseUrl')}${projectPath}/${fileId}/content?token=local-test`,
          target: image,
        }],
      });
      req.reply({ body: { session_id: session, manifest_id: manifestId } });
    }).as('nativeSession');
    cy.intercept('GET', `/neon/samples/manifests/native-${run}-*.jsonld`, (req) => {
      const session = new URL(req.url).pathname.split('/').pop().replace('.jsonld', '');
      expect(manifests.has(session), 'manifest created by edit-session API').to.equal(true);
      req.reply({ body: manifests.get(session) });
    });
    cy.intercept('GET', `${projectPath}/${fileId}/content*`, (req) => {
      server.loads.push(server.content);
      req.reply({ headers: { 'content-type': 'application/xml' }, body: server.content });
    }).as('nativeLoad');
    cy.intercept('PUT', `${projectPath}/${fileId}/content*`, (req) => {
      expect(req.body, 'Neon saves serialized MEI').to.be.a('string');
      assertMei(req.body, false);
      server.content = req.body;
      server.events.push('put');
      server.saveResponseCompleted = false;
      req.on('after:response', () => { server.saveResponseCompleted = true; });
      req.reply({ delay: 75, body: { ok: true } });
    }).as('nativeSave');
    cy.intercept('PATCH', `${projectPath}/${fileId}`, (req) => {
      expect(req.body).to.deep.equal({ corrected: true });
      expect(server.saveResponseCompleted, 'MEI PUT response completed before corrected PATCH')
        .to.equal(true);
      server.corrected = true;
      server.events.push('patch');
      req.reply({ body: { ok: true } });
    }).as('nativeDone');

    observeNeonRenderer();
    cy.visit(pagePath);
    cy.wait('@nativeSession');
    editorElement('#notation-type-label').should('contain.text', 'Hufnagel');
    expectConnected(true);
    expectSaved(true);
    editorElement('#selByNc').click({ force: true }).should('have.class', 'is-active');

    toggleConnection();
    expectConnected(false);
    expectCurrentMei(false);
    expectSaved(false);

    toggleConnection();
    expectConnected(true);
    expectCurrentMei(true);
    expectSaved(false);

    // Finish in a state different from the original file. Seeing that state
    // after reopening proves the edit was saved, not just the original MEI.
    toggleConnection();
    expectConnected(false);
    expectCurrentMei(false);

    cy.contains('button', /^Mark Done$/).click();
    cy.wait('@nativeSave').then(({ request }) => assertMei(request.body, false));
    cy.wait('@nativeDone');
    expectSaved(true);
    cy.contains('button', fileName).should('be.disabled');
    cy.get('#test-events').should('contain.text', `corrected:${fileId}`);

    cy.then(() => {
      expect(server.events, 'save precedes completion').to.deep.equal(['put', 'patch']);
      const loadsBeforeReopen = server.loads.length;
      // Reopening the real component obtains a fresh session, so its current
      // MEI must come from GET rather than the old iframe or a cached manifest.
      cy.visit(pagePath);
      cy.wait('@nativeSession');
      editorElement('#notation-type-label').should('contain.text', 'Hufnagel');
      expectConnected(false);
      expectCurrentMei(false);
      expectSaved(true);
      cy.then(() => {
        expect(server.loads.length, 'reopen fetched the saved MEI').to.be.greaterThan(loadsBeforeReopen);
        server.loads.slice(loadsBeforeReopen).forEach((xml) => {
          expect(xml).to.equal(server.content);
          assertMei(xml, false);
        });
        expect(server.events, 'reopening does not write').to.deep.equal(['put', 'patch']);
      });
    });
  });
});
