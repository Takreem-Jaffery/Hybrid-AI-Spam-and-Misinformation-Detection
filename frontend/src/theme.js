export const THEMES = {
  light: {
    '--bg-base': '#f4f6f8',
    '--surface': '#ffffff',
    '--surface-raised': '#ffffff',
    '--surface-input': '#fbfcfd',
    '--surface-subtle': '#eef2f5',
    '--text-primary': '#17232e',
    '--text-secondary': '#354554',
    '--text-muted': '#5b6b78',
    '--text-faint': '#697986',
    '--border': '#cbd3da',
    '--border-strong': '#aebbc5',
    '--accent': '#087f8c',
    '--accent-soft': '#e5f3f4',
    '--accent-contrast': '#ffffff',
    '--focus': '#087f8c',
    '--shadow': '0 8px 26px rgba(20, 38, 52, .055)',
  },
  dark: {
    '--bg-base': '#111820',
    '--surface': '#19232d',
    '--surface-raised': '#202c37',
    '--surface-input': '#121b24',
    '--surface-subtle': '#222f3a',
    '--text-primary': '#f3f6f8',
    '--text-secondary': '#d0d9df',
    '--text-muted': '#b1bec8',
    '--text-faint': '#9cabb6',
    '--border': '#3b4a57',
    '--border-strong': '#586977',
    '--accent': '#60c6c1',
    '--accent-soft': '#203a3d',
    '--accent-contrast': '#102324',
    '--focus': '#84e1dc',
    '--shadow': '0 8px 26px rgba(0, 0, 0, .18)',
  },
}

export function applyTheme(mode) {
  document.documentElement.dataset.theme = mode
  document.documentElement.style.colorScheme = mode
  Object.entries(THEMES[mode]).forEach(([key, value]) => {
    document.documentElement.style.setProperty(key, value)
  })
}
