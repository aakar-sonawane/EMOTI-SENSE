import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const KEY = 'emotisense.history.v1'
const MAX = 500
const HistoryContext = createContext(null)

function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || '[]')
    return Array.isArray(raw) ? raw : []
  } catch {
    return []
  }
}

export function historyToCsv(items) {
  const esc = (v) => {
    const s = v === null || v === undefined ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const head = ['time', 'source', 'label', 'confidence_percent', 'valence', 'detail']
  const rows = items.map((h) => [
    new Date(h.ts).toISOString(), h.source, h.label,
    h.confidence == null ? '' : (h.confidence * 100).toFixed(1),
    h.valence == null ? '' : h.valence.toFixed(3), h.detail || '',
  ])
  return [head, ...rows].map((r) => r.map(esc).join(',')).join('\n')
}

export function HistoryProvider({ children }) {
  const [items, setItems] = useState(load)
  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(items)) } catch { /* storage full or blocked */ }
  }, [items])
  const addHistory = useCallback((entry) => {
    setItems((prev) => [...prev, { ts: Date.now(), detail: '', confidence: null, valence: null, ...entry }].slice(-MAX))
  }, [])
  const clearHistory = useCallback(() => setItems([]), [])
  const value = useMemo(() => ({ items, addHistory, clearHistory }), [items, addHistory, clearHistory])
  return <HistoryContext.Provider value={value}>{children}</HistoryContext.Provider>
}

export function useHistory() {
  const ctx = useContext(HistoryContext)
  if (!ctx) throw new Error('useHistory must be used inside <HistoryProvider>')
  return ctx
}
