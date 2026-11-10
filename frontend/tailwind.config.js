/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: { ink: '#0b0d18', panel: '#151a2e', line: '#2d304b', accent: '#6c5ce7', accent2: '#4b8bdb' },
      fontFamily: { sans: ['"DM Sans"', 'system-ui', 'sans-serif'], display: ['"Space Grotesk"', 'system-ui', 'sans-serif'] },
    },
  },
  plugins: [],
}
