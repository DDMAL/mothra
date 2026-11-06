import { observeNeonRenderer, waitForNeonRenderer } from './helpers/renderer';

const projectPath = '/api/projects/215/mei';
const pagePath = '/tests/neon-editor/index.html';
const names = { a: 'Square-A.mei', b: 'Hufnagel-B.mei' };

function editorElement(selector) {
  return cy.get('iframe[title^="Neon editor - "]')
    .should(($frames) => {
      const frame = $frames[0];
      expect(frame.contentDocument?.URL, 'current iframe navigation loaded').to.equal(frame.src);
      expect(frame.contentDocument?.querySelector(selector), `${selector} in loaded iframe`).to.exist;
    })
    .then(($frames) => cy.wrap($frames[0].contentDocument).find(selector));
}

function expectEditor(id, notation) {
  cy.get('iframe').should('have.attr', 'title', `Neon editor - ${names[id]}`);
  editorElement('#notation-type-label').should('contain.text', notation);
  editorElement('svg.active-page .nc').should('have.length.greaterThan', 0);
}

function expectRendererIdle(notation) {
  waitForNeonRenderer().then((mei) => assertNotation(mei, notation));
}

function expectSaved(saved = true) {
  editorElement('#file-saved').should('have.attr', 'alt',
    saved ? 'Your work is saved' : 'You have unsaved work');
}

function selectNotation(notation) {
  editorElement('svg.active-page').then(($before) => {
    const before = $before[0];
    editorElement('#notation-type-button').click();
    editorElement(`#notation-type-${notation.toLowerCase()}`).click();
    editorElement('#notation-type-label').should('contain.text', notation);
    // The label updates before the asynchronous encoding/font switch.
    // Wait for its rendered result before testing Save, not an arbitrary delay.
    editorElement('svg.active-page').should(($after) => {
      expect($after[0], 'notation change rendered').not.to.equal(before);
    });
    expectRendererIdle(notation.toLowerCase());
    expectSaved(false);
  });
}

function saveWithMenu() {
  // Exercise Neon's own Save control, which is inside its hover menu.
  editorElement('#save').click({ force: true });
}

function goTo(id) {
  cy.contains('button', names[id]).click();
}

function assertNotation(xml, notation) {
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  expect(doc.querySelector('parsererror'), 'valid saved MEI').to.equal(null);
  const staffDefs = [...doc.querySelectorAll('staffDef')];
  expect(staffDefs, 'saved staff definitions').not.to.be.empty;
  staffDefs.forEach((staff) => {
    expect(staff.getAttribute('notationtype')).to.equal(`neume.${notation}`);
  });
  expect(doc.querySelectorAll('nc').length, 'saved musical content').to.be.greaterThan(0);
}

describe('Mothra with its real embedded Neon editor', () => {
  let server;

  beforeEach(() => {
    server = {
      content: {}, writes: [], corrected: [], events: [],
      loads: { a: [], b: [] }, sessions: { a: [], b: [] },
      completedSaves: { a: false, b: false },
    };
    cy.task('readNeonFixture', 'square').then((xml) => {
      // Current Mothra declares Square explicitly; the historical Neon
      // sample predates those values and otherwise bypasses the old shim.
      server.content.a = xml.replace('notationtype="neume"', 'notationtype="neume.square"');
    });
    cy.task('readNeonFixture', 'hufnagel').then((xml) => { server.content.b = xml; });
    cy.then(() => {
      // A fresh manifest identity for each case prevents unrelated settings
      // from leaking in. Within a case ordinary navigation retains identity.
      const run = `${Date.now()}-${Cypress._.random(1000000)}`;
      const image = `${Cypress.config('baseUrl')}/__test__/page.svg`;
      const manifests = new Map();
      let sessionNumber = 0;
      const files = ['a', 'b'].map((id) => ({
        id, name: names[id], xmlContent: server.content[id], corrected: false,
      }));
      cy.intercept('GET', '/__test__/files', { body: { files } });
      cy.intercept('GET', '/__test__/page.svg', {
        headers: { 'content-type': 'image/svg+xml' },
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="4872" height="6496"><rect width="100%" height="100%" fill="white"/></svg>',
      });
      cy.intercept('GET', `/neon/samples/manifests/test-${run}-*.jsonld`, (req) => {
        const session = new URL(req.url).pathname.split('/').pop().replace('.jsonld', '');
        expect(manifests.has(session), 'manifest created by edit-session API').to.equal(true);
        req.reply({ body: manifests.get(session) });
      });
      ['a', 'b'].forEach((id) => {
        const body = `${Cypress.config('baseUrl')}${projectPath}/${id}/content?token=local-test`;
        cy.intercept('POST', `${projectPath}/${id}/edit-session`, (req) => {
          // Mothra issues a fresh session + manifest on every POST. This also
          // models a renewed session after reopening the editor component.
          const session = `test-${run}-${id}-${++sessionNumber}`;
          server.sessions[id].push(session);
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
            title: names[id],
            image,
            mei_annotations: [{
              id: `urn:uuid:annotation-${session}`,
              type: 'Annotation', body, target: image,
            }],
          });
          req.reply({ body: { session_id: session, manifest_id: manifestId } });
        }).as(`session${id}`);
        cy.intercept('GET', `${projectPath}/${id}/content*`, (req) => {
          server.loads[id].push(server.content[id]);
          req.reply({
            headers: { 'content-type': 'application/xml' },
            body: server.content[id],
          });
        }).as(`load${id}`);
        cy.intercept('PUT', `${projectPath}/${id}/content*`, (req) => {
          expect(req.body, 'Neon sends raw MEI').to.be.a('string');
          server.content[id] = req.body;
          server.writes.push({ id, xml: req.body });
          server.events.push(`put:${id}`);
          server.completedSaves[id] = false;
          req.on('after:response', () => { server.completedSaves[id] = true; });
          req.reply({ delay: 75, body: { ok: true } });
        }).as(`save${id}`);
        cy.intercept('PATCH', `${projectPath}/${id}`, (req) => {
          expect(req.body).to.deep.equal({ corrected: true });
          expect(server.completedSaves[id], 'MEI PUT response completed before corrected PATCH')
            .to.equal(true);
          server.corrected.push(id);
          server.events.push(`patch:${id}`);
          req.reply({ body: { ok: true } });
        }).as(`done${id}`);
      });
    });
    observeNeonRenderer();
    cy.visit(pagePath);
    cy.wait(['@sessiona', '@sessionb']);
  });

  function reopenAndExpectSaved(id, notation) {
    cy.then(() => {
      const loadsBefore = server.loads[id].length;
      const sessionsBefore = [...server.sessions[id]];
      cy.reload();
      cy.wait(['@sessiona', '@sessionb']);
      if (id === 'b') goTo('b');
      expectEditor(id, notation);
      expectRendererIdle(notation.toLowerCase());
      expectSaved();
      cy.then(() => {
        const sessionsAfter = server.sessions[id];
        expect(sessionsAfter.length, 'reopen creates a new edit session')
          .to.be.greaterThan(sessionsBefore.length);
        expect(sessionsBefore).not.to.include(sessionsAfter[sessionsAfter.length - 1]);
        expect(server.loads[id].length, 'new session fetches saved MEI')
          .to.be.greaterThan(loadsBefore);
        server.loads[id].slice(loadsBefore).forEach((xml) => {
          expect(xml, 'GET returns the MEI sent by PUT').to.equal(server.content[id]);
          assertNotation(xml, notation.toLowerCase());
        });
        expect(server.writes, 'reopen does not write').to.have.length(1);
      });
    });
  }

  it('initializes Square and Hufnagel from their own MEI without creating edits', () => {
    expectEditor('a', 'Square');
    expectSaved();
    goTo('b');
    expectEditor('b', 'Hufnagel');
    expectSaved();
    goTo('a');
    expectEditor('a', 'Square');
    expectSaved();
    cy.then(() => expect(server.writes).to.deep.equal([]));
  });

  it('saves Square to Hufnagel through the Save menu, page navigation, and a new session', () => {
    expectEditor('a', 'Square');
    expectRendererIdle('square');
    selectNotation('Hufnagel');
    saveWithMenu();
    cy.wait('@savea').then(({ request }) => assertNotation(request.body, 'hufnagel'));
    expectSaved();
    goTo('b');
    expectEditor('b', 'Hufnagel');
    expectSaved();
    goTo('a');
    expectEditor('a', 'Hufnagel');
    expectSaved();
    cy.then(() => {
      assertNotation(server.content.a, 'hufnagel');
      expect(server.writes).to.have.length(1);
    });
    reopenAndExpectSaved('a', 'Hufnagel');
  });

  it('saves Hufnagel to Square through the Save menu, page navigation, and a new session', () => {
    expectEditor('a', 'Square');
    goTo('b');
    expectEditor('b', 'Hufnagel');
    expectRendererIdle('hufnagel');
    selectNotation('Square');
    saveWithMenu();
    cy.wait('@saveb').then(({ request }) => assertNotation(request.body, 'square'));
    expectSaved();
    goTo('a');
    expectEditor('a', 'Square');
    goTo('b');
    expectEditor('b', 'Square');
    expectSaved();
    cy.then(() => {
      assertNotation(server.content.b, 'square');
      expect(server.writes).to.have.length(1);
    });
    reopenAndExpectSaved('b', 'Square');
  });

  it('cancels or confirms unsaved navigation without writing to the content API', () => {
    expectEditor('a', 'Square');
    selectNotation('Hufnagel');
    goTo('b');
    cy.contains('leave without saving?').should('be.visible');
    cy.contains('button', 'cancel').click();
    expectEditor('a', 'Hufnagel');
    expectSaved(false);
    goTo('b');
    cy.contains('button', 'leave anyway').click();
    expectEditor('b', 'Hufnagel');
    expectSaved();
    goTo('a');
    expectEditor('a', 'Square');
    expectSaved();
    cy.then(() => expect(server.writes).to.deep.equal([]));
  });

  it('Mark Done saves the actual edited MEI before PATCH and advances to the next file', () => {
    expectEditor('a', 'Square');
    expectRendererIdle('square');
    selectNotation('Hufnagel');
    cy.contains('button', 'Mark Done & Next').click();
    cy.wait('@savea').then(({ request }) => assertNotation(request.body, 'hufnagel'));
    cy.wait('@donea');
    expectEditor('b', 'Hufnagel');
    expectSaved();
    cy.contains('button', names.a).should('be.disabled');
    cy.get('#test-events').should('contain.text', 'corrected:a');
    cy.then(() => {
      expect(server.events).to.deep.equal(['put:a', 'patch:a']);
      expect(server.corrected).to.deep.equal(['a']);
      assertNotation(server.content.a, 'hufnagel');
    });
  });
});
