import { COLORS, EMOJI, pct } from './constants'

export function Card({ title, children, className = '' }) {
  return (
    <section className={`rounded-2xl border border-line bg-panel/90 p-4 sm:p-5 ${className}`}>
      {title && <h2 className="mb-3 font-display text-lg font-semibold text-white">{title}</h2>}
      {children}
    </section>
  )
}

export function Button({ children, variant = 'primary', className = '', ...props }) {
  const base = 'min-h-[48px] rounded-xl px-5 text-base font-semibold transition active:scale-[0.98] disabled:opacity-50 disabled:active:scale-100'
  const styles = variant === 'primary'
    ? 'bg-gradient-to-r from-accent to-accent2 text-white shadow-lg shadow-accent/20'
    : 'border border-line bg-white/5 text-slate-100 hover:bg-white/10'
  return <button type="button" className={`${base} ${styles} ${className}`} {...props}>{children}</button>
}

export function ErrorBanner({ message }) {
  if (!message) return null
  return <div role="alert" className="mb-3 rounded-xl border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">{message}</div>
}

export function Notice({ children, tone = 'info' }) {
  const tones = { info: 'border-accent2/40 bg-accent2/10 text-sky-100', warn: 'border-amber-500/40 bg-amber-500/10 text-amber-100' }
  return <div className={`mb-3 rounded-xl border p-3 text-sm ${tones[tone]}`}>{children}</div>
}

export function Spinner({ label = 'Working…' }) {
  return (
    <div role="status" className="flex items-center gap-3 text-sm text-slate-300">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-500 border-t-accent2" />
      {label}
    </div>
  )
}

export function BigEmotion({ label, confidence, note }) {
  return (
    <div className="rounded-2xl border border-line bg-black/20 p-4 text-center">
      <div className="text-5xl" aria-hidden="true">{EMOJI[label] || '🙂'}</div>
      <div className="mt-1 font-display text-2xl font-bold" style={{ color: COLORS[label] || '#fff' }}>{label}</div>
      {confidence != null && <div className="text-sm text-slate-300">confidence {pct(confidence)}</div>}
      {note && <div className="mt-1 text-xs text-slate-400">{note}</div>}
    </div>
  )
}

export function ProbBars({ labels, probs }) {
  const top = probs.indexOf(Math.max(...probs))
  return (
    <ul className="space-y-1.5" aria-label="Class probabilities">
      {labels.map((l, i) => (
        <li key={l} className="flex items-center gap-2 text-sm">
          <span className="w-20 shrink-0 text-slate-300">{l}</span>
          <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-white/10">
            <div className="h-full rounded-full transition-all duration-300"
                 style={{ width: `${Math.max(probs[i] * 100, 0.5)}%`, background: COLORS[l] || '#6c5ce7', opacity: i === top ? 1 : 0.55 }} />
          </div>
          <span className="w-12 shrink-0 text-right tabular-nums text-slate-300">{pct(probs[i])}</span>
        </li>
      ))}
    </ul>
  )
}

export function FilePick({ label, accept, capture, onFile, testId, variant = 'ghost' }) {
  return (
    <label className={`inline-flex min-h-[48px] cursor-pointer items-center justify-center rounded-xl px-5 text-base font-semibold ${
      variant === 'primary' ? 'bg-gradient-to-r from-accent to-accent2 text-white' : 'border border-line bg-white/5 text-slate-100 hover:bg-white/10'}`}>
      {label}
      <input data-testid={testId} type="file" accept={accept} capture={capture} className="sr-only"
             onChange={(e) => { const f = e.target.files && e.target.files[0]; if (f) onFile(f); e.target.value = '' }} />
    </label>
  )
}

export function LineChart({ values, color = '#6c5ce7', label, height = 56 }) {
  const W = 300
  const nums = values.filter((v) => v != null && Number.isFinite(v))
  if (nums.length < 2) return <p className="text-xs text-slate-500">Not enough data for a chart.</p>
  const lo = Math.min(...nums), hi = Math.max(...nums), span = hi - lo || 1
  const x = (i) => (i / (values.length - 1)) * W
  const y = (v) => height - 4 - ((v - lo) / span) * (height - 8)
  const segments = []
  let current = []
  values.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) { if (current.length) segments.push(current); current = [] } else current.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`)
  })
  if (current.length) segments.push(current)
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full" style={{ height }} role="img" aria-label={label}>
      {segments.map((seg, i) => <polyline key={i} points={seg.join(' ')} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />)}
    </svg>
  )
}
