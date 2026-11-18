// Observe the real worker without changing its inputs or renderer behavior.
// The notation label and first SVG can appear before font loading is complete.
export function observeNeonRenderer() {
  // Mark Done can remount an iframe and immediately navigate to another file.
  // Serve the observed copy directly, avoiding an upstream response callback
  // on a script request canceled with that discarded iframe.
  return cy.request({ url: '/neon/Neon-gh/editor.js', log: false }).then((response) => {
    const body = `(() => {
      const NativeWorker = window.Worker;
      window.__neonTestMessages = [];
      window.Worker = class extends NativeWorker {
        constructor(...args) {
          super(...args);
          this.addEventListener('message', (event) => {
            const entry = window.__neonTestMessages.find(x => x.id === event.data?.id);
            if (entry) entry.done = true;
          });
        }
        postMessage(message, ...rest) {
          if (message?.id && ['renderData', 'setFont', 'getMEI', 'edit', 'renderToSVG'].includes(message.action)) {
            window.__neonTestMessages.push({...message, done: false});
          }
          return super.postMessage(message, ...rest);
        }
      };
    })();\n` + response.body;
    cy.intercept('GET', '**/Neon-gh/editor.js', {
      statusCode: 200,
      headers: { 'content-type': 'application/javascript', 'cache-control': 'no-store' },
      body,
    });
  });
}

export function waitForNeonRenderer() {
  return cy.get('iframe').its('0.contentWindow').should((win) => {
    const messages = win.__neonTestMessages;
    expect(messages, 'real worker observed').not.to.be.empty;
    expect(messages.filter((message) => !message.done), 'renderer requests completed')
      .to.deep.equal([]);
    expect(messages.some((message) => message.action === 'renderData'), 'page rendered')
      .to.equal(true);
  }).then((win) => {
    const renders = win.__neonTestMessages.filter((message) => message.action === 'renderData');
    return renders[renders.length - 1].mei;
  });
}
