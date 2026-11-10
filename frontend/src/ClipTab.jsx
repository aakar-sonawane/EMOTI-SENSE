import { useState } from 'react'
import { postForm } from './api'
import ClipExplain from './ClipExplain'
import { useHistory } from './history'
import { BigEmotion, Button, Card, ErrorBanner, FilePick, Notice, Spinner } from './ui'

const mb = (n) => `${(n / 1048576).toFixed(1)} MB`

export function ScoreGauge({ score }) {
  const left = ((score + 3) / 6) * 100
  return (
    <div aria-label={`Sentiment score ${score.toFixed(2)} on a scale from -3 to +3`}>
      <div className="relative h-3 rounded-full" style={{ background: 'linear-gradient(90deg,#e5534b,#9aa0b4,#3fb950)' }}>
        <div className="absolute top-1/2 h-6 w-2 -translate-x-1/2 -translate-y-1/2 rounded bg-white shadow" style={{ left: `${left}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-400"><span>−3 very negative</span><span>0</span><span>+3 very positive</span></div>
    </div>
  )
}

export default function ClipTab() {
  const { addHistory } = useHistory()
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [explanation, setExplanation] = useState(null)
  const [explaining, setExplaining] = useState(false)

  const analyse = async () => {
    setBusy(true); setError(''); setResult(null); setExplanation(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await postForm('/api/av', form)
      setResult(res)
      addHistory({ source: 'Audio+Video clip', label: res.label, confidence: null, valence: res.valence, detail: file.name })
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const explain = async () => {
    setExplaining(true); setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      setExplanation(await postForm('/api/av/explain', form))
    } catch (e) {
      setError(e.message)
    } finally {
      setExplaining(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card title="Sentiment of a speech clip (audio + video)">
        <p className="mb-3 text-sm text-slate-300">Record or choose a short clip of one person speaking to the camera. The model uses their voice and their face and returns a sentiment score from −3 to +3 (trained on CMU-MOSI opinion videos).</p>
        <div className="flex flex-wrap items-center gap-3">
          <FilePick label="🎬 Record clip" accept="video/*" capture="user" onFile={setFile} testId="clip-camera" variant="primary" />
          <FilePick label="📁 Choose clip" accept="video/*" onFile={setFile} testId="clip-gallery" />
        </div>
        {file && (
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <span className="text-sm text-slate-300">{file.name} · {mb(file.size)}</span>
            <Button onClick={analyse} disabled={busy}>{busy ? 'Analysing…' : 'Analyse clip'}</Button>
          </div>
        )}
      </Card>
      <ErrorBanner message={error} />
      {busy && <Spinner label="Extracting audio and face features, then running the model (can take 10–30 s)…" />}
      {result && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card title="Result">
            <BigEmotion label={result.label} note={`score ${result.score >= 0 ? '+' : ''}${result.score.toFixed(2)} · ${Math.round(result.ms.total)} ms`} />
            <div className="mt-4"><ScoreGauge score={result.score} /></div>
          </Card>
          <Card title="Details">
            {result.face_rate < 0.5 && <Notice tone="warn">A face was found in only {Math.round(result.face_rate * 100)}% of the sampled frames, so the video branch received blank frames and the score is less reliable.</Notice>}
            <ul className="space-y-1 text-sm text-slate-300">
              <li>Fusion gate: audio {Math.round(result.gate_audio * 100)}% · video {Math.round(result.gate_video * 100)}%</li>
              <li>Face found in {Math.round(result.face_rate * 100)}% of {result.n_frames} sampled frames</li>
              <li>Read the first {result.audio_used_s.toFixed(1)} s of audio (of {result.audio_total_s.toFixed(1)} s) and the first {result.video_used_s.toFixed(1)} s of video (of {result.video_total_s.toFixed(1)} s)</li>
              <li>Features {Math.round(result.ms.features)} ms · model {Math.round(result.ms.model)} ms</li>
            </ul>
            <p className="mt-2 text-xs text-slate-400">Gate weights show what the fusion gate used, not which modality was more informative.</p>
            {!explanation && <Button variant="ghost" className="mt-3 w-full" onClick={explain} disabled={explaining}>{explaining ? 'Working it out…' : 'Why this score?'}</Button>}
          </Card>
        </div>
      )}
      {explaining && <Spinner label="Re-running the model with parts of the clip removed…" />}
      {explanation && <ClipExplain data={explanation} />}
    </div>
  )
}
