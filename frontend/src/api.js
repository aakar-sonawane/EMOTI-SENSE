export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function detailText(data, status) {
  const d = data && data.detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) return d.map((e) => e.msg || JSON.stringify(e)).join('; ')
  return `The server returned an error (${status}).`
}

async function request(path, options) {
  let res
  try {
    res = await fetch(path, options)
  } catch (e) {
    if (e && e.name === 'AbortError') throw e
    throw new ApiError('Cannot reach the server. Is it running, and is this page opened from the same address?', 0)
  }
  let data = null
  try { data = await res.json() } catch { /* not json */ }
  if (!res.ok) throw new ApiError(detailText(data, res.status), res.status)
  return data
}

export const getJSON = (path) => request(path)
export const postJSON = (path, body) =>
  request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
export const postForm = (path, form) => request(path, { method: 'POST', body: form })
export const postFrame = (blob, signal) =>
  request('/api/face/live', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob, signal })

export const getAssistantStatus = () => getJSON('/api/assistant/status')
export const startAssistant = () => request('/api/assistant/start', { method: 'POST' })

/** POST the chat request and call onEvent for every JSON line the server streams back (NDJSON). */
export async function streamChat(payload, onEvent, signal) {
  let res
  try {
    res = await fetch('/api/assistant/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal,
    })
  } catch (e) {
    if (e && e.name === 'AbortError') throw e
    throw new ApiError('Cannot reach the server. Is it running, and is this page opened from the same address?', 0)
  }
  if (!res.ok) {
    let data = null
    try { data = await res.json() } catch { /* not json */ }
    throw new ApiError(detailText(data, res.status), res.status)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const flush = (line) => { const t = line.trim(); if (t) onEvent(JSON.parse(t)) }
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let nl = buffer.indexOf('\n')
    while (nl >= 0) {
      flush(buffer.slice(0, nl))
      buffer = buffer.slice(nl + 1)
      nl = buffer.indexOf('\n')
    }
  }
  flush(buffer)
}
