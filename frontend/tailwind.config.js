import colors from 'tailwindcss/colors';
import plugin from 'tailwindcss/plugin';

// Theme-aware palettes. Every slate/accent/white utility resolves through a CSS variable, so the
// existing class names render in either theme. Light is the default; `html.dark` restores the
// original dark palette.
const STEPS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];
const ACCENTS = ['sky', 'amber', 'rose', 'emerald', 'red', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'teal', 'violet', 'indigo', 'fuchsia', 'pink', 'green'];

const DARK_SLATE = { ...colors.slate, 850: '#131c2e', 950: '#070c18' };
// Light mode ("sandalwood"): page = 950, cards = 900, insets = 850, borders = 700/800, text = 50..500.
const LIGHT_SLATE = {
  50: '#140f10', 100: '#231d1c', 200: '#332b28', 300: '#463c35', 400: '#5f5348', 500: '#7d6f5e',
  600: '#b3a28a', 700: '#d9cab3', 800: '#ebe0cf', 850: '#fbf5ea', 900: '#fffaf2', 950: '#f1e9dc',
};
// Deep purple neutrals for the navbar and the 3D scanner card, in both themes.
const PURPLE_SLATE = {
  50: '#faf8fd', 100: '#f3effa', 200: '#e8e1f5', 300: '#d6ccea', 400: '#b9acd4', 500: '#8f80ad',
  600: '#6b5a8e', 700: '#4c3870', 800: '#3a2858', 850: '#2f1f48', 900: '#2a1b40', 950: '#1f1430',
};
// The former blue/sky accents render as purple.
const ACCENT_SOURCE = { sky: 'violet', blue: 'purple', cyan: 'fuchsia' };
// Accent shades swap ends of the scale (300 <-> 700, 50 <-> 950), so light text on dark becomes dark on light.
const mirror = (step) => ({ 50: 950, 950: 50 })[step] ?? 1000 - step;

const rgb = (hex) => {
  const n = parseInt(hex.replace('#', ''), 16);
  return `${(n >> 16) & 255} ${(n >> 8) & 255} ${n & 255}`;
};
const varScale = (name, steps) =>
  Object.fromEntries(steps.map((s) => [s, `rgb(var(--c-${name}-${s}) / <alpha-value>)`]));

const darkVars = { '--c-white': rgb('#ffffff') };
const lightVars = { '--c-white': rgb('#2a1f33') };
const purpleVars = { '--c-white': rgb('#ffffff') };
for (const s of [...STEPS, 850]) {
  darkVars[`--c-slate-${s}`] = rgb(DARK_SLATE[s]);
  lightVars[`--c-slate-${s}`] = rgb(LIGHT_SLATE[s]);
  purpleVars[`--c-slate-${s}`] = rgb(PURPLE_SLATE[s]);
}
for (const name of ACCENTS) {
  const source = colors[ACCENT_SOURCE[name] || name];
  for (const s of STEPS) {
    darkVars[`--c-${name}-${s}`] = rgb(source[s]);
    lightVars[`--c-${name}-${s}`] = rgb(source[mirror(s)]);
    purpleVars[`--c-${name}-${s}`] = rgb(source[s]);
  }
}

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0284c7', // Primary medical cyan/blue
          600: '#0369a1',
          700: '#075985',
          800: '#0c4a6e',
          900: '#082f49',
        },
        dr: {
          grade0: '#10b981', // No DR - Emerald
          grade1: '#3b82f6', // Mild - Blue
          grade2: '#f59e0b', // Moderate - Amber
          grade3: '#f97316', // Severe - Orange
          grade4: '#ef4444', // Proliferative - Red
        },
        white: 'rgb(var(--c-white) / <alpha-value>)',
        slate: varScale('slate', [...STEPS, 850]),
        ...Object.fromEntries(ACCENTS.map((name) => [name, varScale(name, STEPS)])),
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        heading: ['Plus Jakarta Sans', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [
    plugin(({ addBase }) => {
      addBase({
        ':root': { ...lightVars, colorScheme: 'light' },
        ':root.dark': { ...darkVars, colorScheme: 'dark' },
        // A region that always uses the deep-purple palette (navbar, 3D scanner card).
        '.theme-purple-island': { ...purpleVars, color: 'rgb(var(--c-slate-100))' },
      });
    }),
  ],
}
