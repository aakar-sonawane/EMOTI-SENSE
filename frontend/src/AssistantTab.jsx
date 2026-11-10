import { useCallback, useEffect, useRef, useState } from 'react'
import { getAssistantStatus, startAssistant, streamChat } from './api'
import { EMOJI, pct } from './constants'
import { useHistory } from './history'
import { Button, Card, ErrorBanner, Notice, Spinner } from './ui'

const STYLES = ['Supportive', 'Neutral', 'Professional']
const FACE_FRESH_MS = 120000   // a camera emotion older than 2 minutes is not shared with the assistant
const POLL_MS = 1500           // while Ollama is starting / loading the model, ask the server again this often
const MAX_POLLS = 80

function MoodLine({ values }) {
  if (values.length < 2) return null
  const W = 300, H = 56
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * W},${H / 2 - v * (H / 2 - 4)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-14 w-full" role="img" aria-label="Mood of your messages">
      <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="#2d304b" strokeDasharray="4 4" />
      <polyline points={pts} fill="none" stroke="#6c5ce7" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  )
}

export default function AssistantTab() {
  const { items, addHistory } = useHistory()
  const [status, setStatus] = useState(null)
  const [model, setModel] = useState('')
  const [style, setStyle] = useState('Supportive')
  const [analyse, setAnalyse] = useState(true)
  const [useFace, setUseFace] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const abortRef = useRef(null)
  const endRef = useRef(null)

  const pollsRef = useRef(0)
  const loadStatus = useCallback(async (initial = false) => {
    if (initial) { setStatus(null); pollsRef.current = 0 }
    try {
      const s = await getAssistantStatus()
      setStatus(s)
      setModel((m) => m || s.preferred || '')
    } catch (e) {
      setStatus({ available: false, models: [], error: e.message, url: '' })
    }
  }, [])
  useEffect(() => { loadStatus(true) }, [loadStatus])
  useEffect(() => {
    if (!status || !(status.starting || status.warming) || pollsRef.current >= MAX_POLLS) return undefined
    const id = setTimeout(() => { pollsRef.current += 1; loadStatus() }, POLL_MS)
    return () => clearTimeout(id)
  }, [status, loadStatus])
  const checkAgain = async () => {
    try { await startAssistant() } catch { /* the status call below shows what is wrong */ }
    loadStatus(true)
  }
  useEffect(() => { if (endRef.current && endRef.current.scrollIntoView) endRef.current.scrollIntoView({ block: 'end' }) }, [messages])
  useEffect(() => () => { if (abortRef.current) abortRef.current.abort() }, [])

  const latestFace = () => {
    const hit = [...items].reverse().find((h) => (h.source === 'Live camera' || h.source === 'Photo') && Date.now() - h.ts <= FACE_FRESH_MS)
    return hit ? { label: hit.label, confidence: hit.confidence ?? 0.5, age_s: Math.round((Date.now() - hit.ts) / 1000) } : null
  }
  const face = latestFace()

  const patch = (index, change) => setMessages((prev) => prev.map((m, i) => (i === index ? { ...m, ...change } : m)))

  const send = async () => {
    const text = input.trim()
    if (!text || streaming || !model) return
    const history = [...messages, { role: 'user', content: text }]
    const userIndex = history.length - 1, botIndex = history.length
    setMessages([...history, { role: 'assistant', content: '', pending: true }])
    setInput(''); setError(''); setNotice(''); setStreaming(true)
    const controller = new AbortController()
    abortRef.current = controller
    let reply = ''
    try {
      await streamChat({
        model, style, temperature: 0.7, analyse, face_emotion: useFace ? latestFace() : null,
        messages: history.map(({ role, content }) => ({ role, content })),
      }, (ev) => {
        if (ev.type === 'sentiment') {
          patch(userIndex, { sentiment: ev })
          addHistory({ source: 'Assistant', label: ev.label, confidence: ev.confidence, valence: ev.valence, detail: text.slice(0, 60) })
        } else if (ev.type === 'crisis') patch(botIndex, { crisis: ev.note })
        else if (ev.type === 'notice') setNotice(ev.message)
        else if (ev.type === 'delta') { reply += ev.text; patch(botIndex, { content: reply }) }
        else if (ev.type === 'error') setError(ev.message)
      }, controller.signal)
    } catch (e) {
      if (!e || e.name !== 'AbortError') setError(e.message)
    } finally {
      setMessages((prev) => prev.flatMap((m, i) => {
        if (i !== botIndex) return [m]
        const { pending, ...rest } = m   // eslint-disable-line no-unused-vars
        return rest.content || rest.crisis ? [rest] : []
      }))
      setStreaming(false)
      abortRef.current = null
    }
  }

  const stop = () => { if (abortRef.current) abortRef.current.abort() }
  const reset = () => { stop(); setMessages([]); setError(''); setNotice('') }
  const onKey = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }

  const moods = messages.filter((m) => m.role === 'user' && m.sentiment).map((m) => m.sentiment.valence)
  const k = Math.max(1, Math.min(3, Math.floor(moods.length / 2)))
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length

  if (!status) return <Spinner label="Looking for Ollama on this computer…" />
  if (!status.available && status.starting) {
    return (
      <Card title="Emotion-aware assistant">
        <Spinner label={status.phase_message || 'Starting Ollama…'} />
        <p className="mt-2 text-sm text-slate-400">Ollama is started for you, this takes a few seconds. You do not need to run anything.</p>
      </Card>
    )
  }
  if (!status.available) {
    return (
      <Card title="Emotion-aware assistant">
        <ErrorBanner message={status.error} />
        <p className="mb-2 text-sm text-slate-300">The assistant runs on your own laptop through Ollama, so nothing is sent to the internet. The server tries to start Ollama by itself; if that did not work, start it like this:</p>
        <pre className="mb-3 overflow-x-auto rounded-xl bg-black/40 p-3 text-sm text-slate-200">{'ollama serve\nollama pull qwen3:8b'}</pre>
        <Button onClick={checkAgain}>Check again</Button>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {status.warming && <Notice>{status.phase_message || 'Loading the model into memory…'} You can already type, the first answer may just take a little longer.</Notice>}
      <Notice>Sentiment estimates are automatic and can be wrong (the sentiment model understands English only). This assistant is a demo, not a therapist or a medical tool.</Notice>
      <Card title="Settings">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-sm text-slate-300">Local model
            <select value={model} onChange={(e) => setModel(e.target.value)} className="mt-1 w-full rounded-xl border border-line bg-black/30 p-3 text-white">
              {status.models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
          <label className="text-sm text-slate-300">Reply style
            <select value={style} onChange={(e) => setStyle(e.target.value)} className="mt-1 w-full rounded-xl border border-line bg-black/30 p-3 text-white">
              {STYLES.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
        </div>
        <label className="mt-3 flex min-h-[44px] items-center gap-3 text-sm text-slate-200">
          <input type="checkbox" checked={analyse} onChange={(e) => setAnalyse(e.target.checked)} className="h-5 w-5 accent-[#6c5ce7]" />
          Estimate how each of my messages feels (BERT–RoBERTa) and tell the assistant
        </label>
        <label className="flex min-h-[44px] items-center gap-3 text-sm text-slate-200">
          <input type="checkbox" checked={useFace} onChange={(e) => setUseFace(e.target.checked)} className="h-5 w-5 accent-[#6c5ce7]" />
          <span>Also share my latest face emotion from the Live or Photo tab
            <span className="block text-xs text-slate-400">{face ? `Right now: ${EMOJI[face.label] || ''} ${face.label} (${face.age_s} s ago)` : 'No face emotion in the last 2 minutes. Use the Live or Photo tab first.'}</span>
          </span>
        </label>
      </Card>

      <ErrorBanner message={error} />
      {notice && <Notice tone="warn">{notice}</Notice>}

      <Card title="Conversation">
        {messages.length === 0 && <p className="text-sm text-slate-400">Say something, for example “I have my viva tomorrow and I am nervous”. English, Hindi and Hinglish all work for the chat itself.</p>}
        <ul className="space-y-3" aria-live="polite">
          {messages.map((m, i) => (
            <li key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[88%] rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed ${m.role === 'user' ? 'bg-gradient-to-r from-accent to-accent2 text-white' : 'border border-line bg-black/30 text-slate-100'}`}>
                {m.crisis && <p role="note" className="mb-2 rounded-lg border border-amber-400/40 bg-amber-400/10 p-2 text-sm text-amber-100">{m.crisis}</p>}
                <p className="whitespace-pre-wrap">{m.content}{m.pending && <span className="ml-1 inline-block h-4 w-1.5 animate-pulse bg-slate-300 align-middle" aria-label="typing" />}</p>
                {m.sentiment && <p className="mt-1 text-xs text-white/80">{EMOJI[m.sentiment.label] || ''} {m.sentiment.label} · {pct(m.sentiment.confidence)}</p>}
              </div>
            </li>
          ))}
        </ul>
        <div ref={endRef} />
        <div className="mt-4 flex items-end gap-2">
          <label htmlFor="chat-input" className="sr-only">Your message</label>
          <textarea id="chat-input" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKey} rows={2} maxLength={2000}
                    placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
                    className="min-h-[52px] flex-1 rounded-xl border border-line bg-black/30 p-3 text-base text-white placeholder:text-slate-500 focus:border-accent focus:outline-none" />
          {streaming ? <Button variant="ghost" onClick={stop}>Stop</Button> : <Button onClick={send} disabled={!input.trim() || !model}>Send</Button>}
        </div>
        <div className="mt-3"><Button variant="ghost" onClick={reset}>New conversation</Button></div>
      </Card>

      {moods.length >= 2 && (
        <Card title="Mood of your messages">
          <MoodLine values={moods} />
          <p className="text-sm text-slate-300">
            {moods.length} messages · average {avg(moods) >= 0 ? '+' : ''}{avg(moods).toFixed(2)} · from {avg(moods.slice(0, k)).toFixed(2)} at the start to {avg(moods.slice(-k)).toFixed(2)} now
          </p>
          <p className="mt-1 text-xs text-slate-400">Scale: −1 negative … +1 positive. An automatic estimate from your wording, not a measurement of how you feel.</p>
        </Card>
      )}
    </div>
  )
}
