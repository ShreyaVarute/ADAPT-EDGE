/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0B0F19',
          surface: '#111827',
          card: '#1F2937',
          border: '#374151',
          hover: '#283548'
        },
        cyan: {
          accent: '#00E5FF',
          dim: '#00B4D8'
        },
        indigo: {
          accent: '#6366F1',
          glow: '#4F46E5'
        }
      }
    },
  },
  plugins: [],
}

