import { useState } from 'react'
import { postJSON } from './api'
import { TEXT_LABELS } from './constants'
import { useHistory } from './history'
import { BigEmotion, Button, Card, ErrorBanner, ProbBars, Spinner } from './ui'

const EXAMPLES = [
  'The staff were friendly and the food was excellent.',
  'This was a complete waste of money. I want a refund.',
  'The package arrived on Tuesday.',
]

export function WordHighlights({ words, scores }) {
  const max = Math.max(1e-6, ...scores.map((s) => Math.abs(s)))
  return (
    <p className="leading-9">
      {words.map((w, i) => {
        const alpha = Math.min(Math.abs(scores[i]) / max, 1) * 0.75
        const bg = scores[i] > 0 ? `rgba(63,185,80,${alpha.toFixed(2)})` : `rgba(229,83,75,${alpha.toFixed(2)})`
        return (
          <span key={i} title={scores[i].toFixed(3)} data-testid="word" className="m-0.5 inline-block rounded-md px-1.5" style={{ background: bg }}>{w}</span>
        )
      })}
    </p>
  )
}

export default function TextTab() {
  const { addHistory } = useHistory()
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [explain, setExplain] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const analyse = async () => {
    setBusy('analyse'); setError(''); setExplain(null)
    try {
      const res = await postJSON('/api/text', { text })
      setResult({ ...res, analysed: text })
      addHistory({ source: 'Text', label: res.label, confidence: res.confidence, valence: res.valence, detail: text.slice(0, 60) })
    } catch (e) {
      setResult(null); setError(e.message)
    } finally {
      setBusy('')
    }
  }

  const explainWords = async () => {
    setBusy('explain'); setError('')
    try {
      setExplain(await postJSON('/api/text/explain', { text: result.analysed }))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="space-y-4">
      <Card title="Sentiment of a sentence">
        <label htmlFor="text-input" className="sr-only">Text to analyse</label>
        <textarea id="text-input" value={text} onChange={(e) => setText(e.target.value)} rows={4} maxLength={5000}
                  placeholder="Type or paste a review, comment or sentence (English)…"
                  className="w-full rounded-xl border border-line bg-black/30 p-3 text-base text-white placeholder:text-slate-500 focus:border-accent focus:outline-none" />
        <div className="mt-2 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => setText(ex)} className="rounded-full border border-line px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10">
              {ex.length > 34 ? `${ex.slice(0, 34)}…` : ex}
            </button>
          ))}
        </div>
        <div className="mt-3">
          <Button onClick={analyse} disabled={!text.trim() || busy !== ''} className="w-full sm:w-auto">Analyse sentiment</Button>
        </div>
        <p className="mt-2 text-xs text-slate-400">The model is trained on English text; Hinglish and sarcasm are often misread.</p>
      </Card>
      <ErrorBanner message={error} />
      {busy === 'analyse' && <Spinner label="Running BERT–RoBERTa…" />}
      {result && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card title="Result">
            <BigEmotion label={result.label} confidence={result.confidence} note={`${Math.round(result.ms)} ms`} />
            {result.confidence < 0.5 && <p className="mt-2 text-sm text-amber-200">Low confidence: no class is above 50%.</p>}
            <div className="mt-3"><ProbBars labels={TEXT_LABELS} probs={result.probs} /></div>
          </Card>
          <Card title="Which words mattered?">
            {!explain && (
              <>
                <p className="mb-3 text-sm text-slate-300">Each word is removed in turn to see how much the prediction depends on it.</p>
                <Button variant="ghost" onClick={explainWords} disabled={busy !== ''}>{busy === 'explain' ? 'Computing…' : 'Explain this prediction'}</Button>
              </>
            )}
            {explain && (
              <>
                <WordHighlights words={explain.words} scores={explain.scores} />
                <p className="mt-2 text-xs text-slate-400">Green supports “{explain.label}”, red works against it.{explain.truncated ? ' Only the first 40 words were analysed.' : ''}</p>
              </>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
