import { useEffect, useState } from 'react'
import { getJSON } from './api'
import { Button, Card, ErrorBanner, Spinner } from './ui'

const Badge = ({ ok, children }) => (
  <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${ok ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300'}`}>{children}</span>
)

export default function SystemTab() {
  const [health, setHealth] = useState(null)
  const [tests, setTests] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let alive = true
    getJSON('/api/health').then((h) => alive && setHealth(h)).catch((e) => alive && setError(e.message))
    return () => { alive = false }
  }, [])

  const runTests = async () => {
    setBusy(true); setError(''); setTests(null)
    try {
      setTests((await getJSON('/api/selftest')).results)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const names = { text: 'Text model (BERT + RoBERTa)', face: 'Face model (VGG16 + ResNet50 + InceptionV3)', av: 'Audio + video model' }
  return (
    <div className="space-y-4">
      <ErrorBanner message={error} />
      <Card title="Models and files">
        {!health && !error && <Spinner label="Checking the server…" />}
        {health && (
          <ul className="space-y-3">
            {Object.entries(health.models).map(([key, m]) => (
              <li key={key} className="text-sm">
                <div className="flex flex-wrap items-center gap-2"><span className="font-semibold">{names[key]}</span><Badge ok={m.found}>{m.found ? 'file found' : 'file missing'}</Badge>{m.loaded && <Badge ok>loaded</Badge>}</div>
                <div className="break-all text-xs text-slate-400">{m.file}</div>
              </li>
            ))}
            <li className="text-sm">
              <div className="flex flex-wrap items-center gap-2"><span className="font-semibold">Face detector</span><Badge ok={health.face_detector.ok}>{health.face_detector.ok ? 'ready' : 'needs fix'}</Badge></div>
              <div className="text-xs text-slate-400">{health.face_detector.detail}</div>
            </li>
            <li className="text-xs text-slate-400">Packages: {Object.entries(health.packages).map(([n, ok]) => `${n} ${ok ? '✓' : '✗'}`).join(' · ')}</li>
          </ul>
        )}
      </Card>
      <Card title="Self-test">
        <p className="mb-3 text-sm text-slate-300">Runs the real models on built-in inputs: a clearly positive and a clearly negative sentence, a face-shape check, and a model-load check. The first run loads the models and can take a minute.</p>
        <Button onClick={runTests} disabled={busy}>{busy ? 'Running…' : 'Run self-test'}</Button>
        {busy && <div className="mt-3"><Spinner label="Loading models and testing…" /></div>}
        {tests && (
          <ul className="mt-3 space-y-2">
            {tests.map((t) => (
              <li key={t.name} className="text-sm"><Badge ok={t.ok}>{t.ok ? 'PASS' : 'FAIL'}</Badge> <span className="font-medium">{t.name}</span><div className="text-xs text-slate-400">{t.detail}</div></li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}
