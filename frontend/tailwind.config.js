/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
      colors: {
        ink: { 950: "#0b0f17", 900: "#111827", 800: "#1a2233", 700: "#263046", 600: "#36425c" },
        brand: { 300: "#a5b4fc", 400: "#818cf8", 500: "#6366f1", 600: "#4f46e5" },
        expert: { id: "#94a3b8", sp: "#f59e0b", blur: "#38bdf8", occ: "#34d399" },
      },
    },
  },
  plugins: [],
};
