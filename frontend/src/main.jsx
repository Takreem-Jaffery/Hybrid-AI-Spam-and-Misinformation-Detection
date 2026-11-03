import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { applyTheme } from './theme.js'
import App from './App.jsx'
import HowItWorks from './HowItWorks.jsx'

applyTheme('light')

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {window.location.pathname === '/how-it-works' ? <HowItWorks /> : <App />}
  </StrictMode>,
)
