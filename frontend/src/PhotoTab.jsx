import { useEffect, useState } from 'react'
import { postForm } from './api'
import { FACE_LABELS } from './constants'
import { useHistory } from './history'
import { BigEmotion, Button, Card, ErrorBanner, FilePick, Notice, ProbBars, Spinner } from './ui'

export default function PhotoTab() {
  const { addHistory } = useHistory()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [crop, setCrop] = useState(true)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [heat, setHeat] = useState(null)
  const [heatBusy, setHeatBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const analyse = async (f, cropFace) => {
    setBusy(true); setError(''); setResult(null); setHeat(null)
    try {
      const form = new FormData()
      form.append('file', f)
      form.append('crop', String(cropFace))
      const res = await postForm('/api/face', form)
      setResult(res)
      addHistory({ source: 'Photo', label: res.label, confidence: res.confidence, valence: res.valence, detail: f.name || 'photo' })
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const onFile = (f) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    analyse(f, crop)
  }

  const showHeatmap = async () => {
    setHeatBusy(true); setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('crop', String(crop))
      setHeat(await postForm('/api/face/heatmap', form))
    } catch (e) {
      setError(e.message)
    } finally {
      setHeatBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card title="Read an expression from a photo">
        <p className="mb-3 text-sm text-slate-300">Take a photo with your camera or pick one from your gallery. The emotion appears right away. These buttons also work on phones without HTTPS.</p>
        <div className="flex flex-wrap items-center gap-3">
          <FilePick label="📷 Take photo" accept="image/*" capture="user" onFile={onFile} testId="photo-camera" variant="primary" />
          <FilePick label="🖼️ Choose photo" accept="image/*" onFile={onFile} testId="photo-gallery" />
          <label className="flex min-h-[48px] items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={crop} onChange={(e) => setCrop(e.target.checked)} className="h-5 w-5 accent-[#6c5ce7]" />
            Detect and crop the face first
          </label>
        </div>
      </Card>
      <ErrorBanner message={error} />
      {busy && <Spinner label="Reading the expression…" />}
      {result && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card title="Result">
            <BigEmotion label={result.label} confidence={result.confidence} note={`${Math.round(result.ms)} ms`} />
            {result.confidence < 0.5 && <Notice tone="warn">Low confidence: no emotion is above 50%. Try better light and a frontal face.</Notice>}
            {result.note && <p className="mt-2 text-xs text-slate-400">{result.note}</p>}
            <div className="mt-3"><ProbBars labels={FACE_LABELS} probs={result.probs} /></div>
          </Card>
          <Card title="What the model saw">
            <div className="grid grid-cols-2 gap-3">
              <figure><img src={result.face} alt="Face used by the model" className="w-full rounded-xl" /><figcaption className="mt-1 text-xs text-slate-400">Face used</figcaption></figure>
              {heat && <figure><img src={heat.overlay} alt="Occlusion heatmap" className="w-full rounded-xl" /><figcaption className="mt-1 text-xs text-slate-400">Red = most important</figcaption></figure>}
            </div>
            {!heat && <Button variant="ghost" className="mt-3 w-full" onClick={showHeatmap} disabled={heatBusy}>{heatBusy ? 'Computing…' : 'Where did the model look?'}</Button>}
            <p className="mt-2 text-xs text-slate-400">The heatmap hides parts of the face one at a time. It shows where the model looks, not why a person feels something.</p>
          </Card>
        </div>
      )}
      {preview && !result && !busy && <img src={preview} alt="Selected" className="max-h-64 rounded-xl" />}
    </div>
  )
}
