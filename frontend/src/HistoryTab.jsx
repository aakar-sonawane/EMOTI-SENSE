import { EMOJI, pct } from './constants'
import { historyToCsv, useHistory } from './history'
import { Button, Card } from './ui'

export function Sparkline({ values }) {
  if (values.length < 2) return null
  const W = 300, H = 60
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * W},${H / 2 - v * (H / 2 - 4)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-16 w-full" role="img" aria-label="Valence over time">
      <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="#2d304b" strokeDasharray="4 4" />
      <polyline points={pts} fill="none" stroke="#6c5ce7" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  )
}

export default function HistoryTab() {
  const { items, clearHistory } = useHistory()
  const download = () => {
    const blob = new Blob([historyToCsv(items)], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `emotisense_history_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }
  const valences = items.filter((h) => h.valence != null).slice(-60).map((h) => h.valence)
  const counts = items.reduce((acc, h) => ({ ...acc, [h.label]: (acc[h.label] || 0) + 1 }), {})
  const rows = [...items].reverse().slice(0, 200)
  return (
    <div className="space-y-4">
      <Card title="Session history">
        {items.length === 0 ? (
          <p className="text-sm text-slate-400">Nothing yet. Predictions from every tab are saved here, on this device only, with a timestamp.</p>
        ) : (
          <>
            <p className="mb-2 text-sm text-slate-300">{items.length} predictions · {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([l, n]) => `${EMOJI[l] || ''} ${l} ${n}`).join(' · ')}</p>
            <p className="mb-1 text-xs text-slate-400">Valence over time (−1 negative … +1 positive; every source is mapped to this scale)</p>
            <Sparkline values={valences} />
            <div className="mt-3 flex flex-wrap gap-3">
              <Button onClick={download}>Download CSV</Button>
              <Button variant="ghost" onClick={clearHistory}>Clear history</Button>
            </div>
          </>
        )}
      </Card>
      {rows.length > 0 && (
        <Card title="Latest">
          <ul className="divide-y divide-line">
            {rows.map((h, i) => (
              <li key={`${h.ts}-${i}`} className="flex items-center justify-between gap-3 py-2 text-sm">
                <span><span aria-hidden="true">{EMOJI[h.label] || '🙂'}</span> {h.label} <span className="text-slate-400">· {h.source}{h.detail ? ` · ${h.detail}` : ''}</span></span>
                <span className="shrink-0 text-xs tabular-nums text-slate-400">{h.confidence != null ? `${pct(h.confidence)} · ` : ''}{new Date(h.ts).toLocaleTimeString()}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}
