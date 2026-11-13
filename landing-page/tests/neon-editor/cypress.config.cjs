const fs = require('node:fs');
const path = require('node:path');

// Use the existing Cypress installation in the Neon submodule.
module.exports = {
  video: false,
  viewportWidth: 1440,
  viewportHeight: 1000,
  defaultCommandTimeout: 30000,
  requestTimeout: 30000,
  responseTimeout: 30000,
  screenshotsFolder: path.join(__dirname, 'artifacts/screenshots'),
  e2e: {
    baseUrl: 'http://127.0.0.1:5178',
    specPattern: path.join(__dirname, '*.cy.js'),
    supportFile: false,
    fixturesFolder: false,
    setupNodeEvents(on) {
      const handleResults = [];
      const fixtureNames = {
        square: 'test.mei',
        hufnagel: 'St_Gall_022r_one_staff.mei',
        mothraHufnagelClivis: path.join(__dirname, 'fixtures/current-mothra-hufnagel-clivis2a.mei'),
      };
      on('task', {
        recordHandleResult(result) {
          handleResults.push(result);
          const directory = path.join(__dirname, 'artifacts');
          fs.mkdirSync(directory, { recursive: true });
          fs.writeFileSync(path.join(directory, 'handle-results.json'),
            JSON.stringify(handleResults, null, 2) + '\n');
          return null;
        },
        readNeonFixture(type) {
          if (!Object.hasOwn(fixtureNames, type)) {
            throw new Error(`Unknown Neon fixture: ${type}`);
          }
          return fs.readFileSync(path.resolve(
            __dirname,
            '../../neon/deployment/server/samples/mei',
            fixtureNames[type],
          ), 'utf8');
        },
      });
    },
  },
};
