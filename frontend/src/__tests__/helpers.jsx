import { render } from '@testing-library/react'
import { vi } from 'vitest'
import { HistoryProvider } from '../history'

export const ok = (data, status = 200) => ({ ok: status < 400, status, json: async () => data })
export const fail = (detail, status = 500) => ({ ok: false, status, json: async () => ({ detail }) })
export const later = (value, ms = 5) => new Promise((resolve) => setTimeout(() => resolve(value), ms))

export function renderWithHistory(ui) {
  return render(<HistoryProvider>{ui}</HistoryProvider>)
}

export function installBrowserStubs() {
  globalThis.URL.createObjectURL = vi.fn(() => 'blob:preview')
  globalThis.URL.revokeObjectURL = vi.fn()
}

/** A fetch Response whose body streams the given NDJSON events in awkward chunks (split in the middle of lines). */
export function ndjsonResponse(events, { chunk = 17, status = 200, delayMs = 0, signal } = {}) {
  const bytes = new TextEncoder().encode(events.map((e) => JSON.stringify(e)).join('\n') + '\n')
  const parts = []
  for (let i = 0; i < bytes.length; i += chunk) parts.push(bytes.slice(i, i + chunk))
  let i = 0
  return {
    ok: status < 400, status, json: async () => ({}),
    body: { getReader: () => ({ read: async () => {
      if (delayMs) await new Promise((r) => setTimeout(r, delayMs))
      if (signal && signal.aborted) throw Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' })   // like a real fetch body
      return i < parts.length ? { value: parts[i++], done: false } : { value: undefined, done: true }
    } }) },
  }
}
