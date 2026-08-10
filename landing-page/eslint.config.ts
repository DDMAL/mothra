import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';
import { defineConfig, globalIgnores } from 'eslint/config';

export default defineConfig([
  // 'neon' is a separate git submodule (its own repo/history) — its lint
  // issues aren't ours to fix here. 'public/neon' is that submodule's own
  // built/bundled editor output, not hand-written source. 'scripts/.venv'
  // is a Python virtualenv that happens to vendor some .js (e.g.
  // matplotlib's web backend) — never meant to be linted by this project's
  // config at all.
  globalIgnores(['dist', 'neon', 'public/neon', 'scripts/.venv']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
]);
