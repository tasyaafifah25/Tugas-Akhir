/** @type {import('tailwindcss').Config} */

const plugin = require("tailwindcss/plugin");

module.exports = {
  content: ["./templates/**/*.html", "./static/**/*.js"],
  theme: {
    extend: {
      colors: {
        primary: "#71B8BA",
        secondary: "#FF9D40",
        main: "#0C0C0C",
      },
    },
  },
  plugins: [
    plugin(function ({ addComponents }) {
      addComponents({
        ".container": {
          width: "100%",
          paddingLeft: "16px",
          paddingRight: "16px",
          "@screen sm": {
            maxWidth: "640px",
            marginLeft: "auto",
            marginRight: "auto",
            paddingLeft: "0",
            paddingRight: "0",
          },
          "@screen md": {
            maxWidth: "700px",
          },
          "@screen lg": {
            maxWidth: "920px",
          },
          "@screen xl": {
            maxWidth: "1120px",
          },
          "@screen 2xl": {
            maxWidth: "1440px",
          },
        },
      });
    }),
  ],
};

