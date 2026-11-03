// All network access lives here. The browser never talks to Groq directly any more: the
// /advice endpoint on the FastAPI server holds the API key.
const BASE = import.meta.env.VITE_API_URL ?? ''

async function post(path, body, timeoutMs) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch {
    throw new Error('Cannot reach the detector API. Is the backend running on port 8000?')
  }
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (HTTP ${res.status})`)
  return data
}

// First request can be slow on CPU while DistilBERT warms up, hence the generous timeout.
export const predict = (message, isForwarded, source) =>
  post('/predict', { message, is_forwarded: isForwarded, source }, 30000)

export const reviewArticle = (title, body) =>
  post('/article/review', { title, body }, 60000)

export const getAdvice = (message, isForwarded, source) =>
  post('/advice', { message, is_forwarded: isForwarded, source }, 30000)
