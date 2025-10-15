/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx,html}',
    // Also scan the sibling `Frontend` folder in case tooling uses that path
    '../Frontend/index.html',
    '../Frontend/src/**/*.{js,jsx,ts,tsx,html}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
