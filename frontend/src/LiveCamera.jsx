import { useCallback, useEffect, useRef, useState } from 'react'
import { postFrame } from './api'
import { COLORS, EMOJI, FACE_LABELS, argmax, faceValence, pct } from './constants'
import { useHistory } from './history'
import { BigEmotion, Button, Card, ErrorBanner, Notice, ProbBars } from './ui'

const GRAB_WIDTH = 320          // frames are downscaled before upload: small = low latency
const SMOOTHING = 0.6           // weight of the previous probabilities (steadier labels)
const LOG_EVERY_MS = 3000       // how often the dominant emotion is written to History
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

export function describeCameraError(e) {
  const name = e && e.name
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Camera permission was denied. Allow the camera for this site in the browser settings and try again.'
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'No camera was found on this device.'
  if (name === 'NotReadableError') return 'The camera is being used by another app. Close it and try again.'
  return `The camera could not be started${e && e.message ? ` (${e.message})` : ''}. Phones only allow the live camera on HTTPS pages; use the Photo tab otherwise.`
}

export default function LiveCamera() {
  const { addHistory } = useHistory()
  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const grabRef = useRef(null)
  const runningRef = useRef(false)
  const runIdRef = useRef(0)   // bumped on every start/stop, so a stale loop can never keep running
  const streamRef = useRef(null)
  const smoothRef = useRef([])
  const lastLogRef = useRef(0)
  const facingRef = useRef('user')
  const controllerRef = useRef(null)
  const [running, setRunning] = useState(false)
  const [facing, setFacing] = useState('user')
  const [error, setError] = useState('')
  const [faces, setFaces] = useState([])
  const [stats, setStats] = useState({ fps: 0, ms: 0 })
  const [aspect, setAspect] = useState(4 / 3)
  const insecure = typeof window !== 'undefined' && window.isSecureContext === false

  const clearOverlay = () => {
    const c = overlayRef.current
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height)
  }

  const draw = useCallback((list) => {
    const canvas = overlayRef.current, video = videoRef.current
    if (!canvas || !video) return
    const W = video.clientWidth, H = video.clientHeight
    if (!W || !H) return
    canvas.width = W
    canvas.height = H
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)
    const mirror = facingRef.current === 'user'
    list.forEach((f) => {
      const [bx, by, bw, bh] = f.box
      const x = (mirror ? 1 - (bx + bw) : bx) * W, y = by * H, w = bw * W, h = bh * H
      const color = COLORS[f.label] || '#6c5ce7'
      ctx.lineWidth = 3
      ctx.strokeStyle = color
      ctx.strokeRect(x, y, w, h)
      const text = `${EMOJI[f.label] || ''} ${f.label} ${pct(f.confidence)}`
      ctx.font = '600 15px system-ui, sans-serif'
      const tw = ctx.measureText(text).width + 12
      ctx.fillStyle = color
      ctx.fillRect(x, Math.max(y - 26, 0), tw, 24)
      ctx.fillStyle = '#0b0d18'
      ctx.fillText(text, x + 6, Math.max(y - 9, 16))
    })
  }, [])

  const handleResult = useCallback((res) => {
    const previous = smoothRef.current
    const next = res.faces.map((f, i) => {
      const prev = previous[i]
      const probs = prev && prev.length === f.probs.length ? prev.map((p, k) => SMOOTHING * p + (1 - SMOOTHING) * f.probs[k]) : f.probs
      const top = argmax(probs)
      return { box: f.box, probs, label: FACE_LABELS[top], confidence: probs[top], valence: faceValence(probs) }
    })
    smoothRef.current = next.map((f) => f.probs)
    setFaces(next)
    draw(next)
    const now = Date.now()
    if (next.length && now - lastLogRef.current > LOG_EVERY_MS) {
      lastLogRef.current = now
      addHistory({ source: 'Live camera', label: next[0].label, confidence: next[0].confidence, valence: next[0].valence })
    }
  }, [draw, addHistory])

  const grab = (video) => {
    if (!grabRef.current) grabRef.current = document.createElement('canvas')
    const canvas = grabRef.current
    canvas.width = GRAB_WIDTH
    canvas.height = Math.max(1, Math.round((GRAB_WIDTH * video.videoHeight) / video.videoWidth))
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.7))
  }

  // Send the next frame only after the previous answer arrived: the newest frame always wins and lag never builds up.
  const loop = useCallback(async (runId) => {
    const alive = () => runningRef.current && runIdRef.current === runId
    let count = 0, windowStart = performance.now()
    while (alive()) {
      const video = videoRef.current
      if (!video || !video.videoWidth) { await sleep(60); continue }
      const blob = await grab(video)
      if (!blob) { await sleep(60); continue }
      if (!alive()) break
      const sentAt = performance.now()
      controllerRef.current = new AbortController()
      try {
        const res = await postFrame(blob, controllerRef.current.signal)
        if (!alive()) break
        setError('')
        handleResult(res)
        count += 1
        const now = performance.now()
        if (now - windowStart >= 1000) {
          setStats({ fps: (count * 1000) / (now - windowStart), ms: now - sentAt })
          count = 0
          windowStart = now
        }
      } catch (e) {
        if (!alive() || (e && e.name === 'AbortError')) break
        setError(e.message)
        await sleep(1000)
      }
    }
  }, [handleResult])

  const stop = useCallback(() => {
    runningRef.current = false
    runIdRef.current += 1
    if (controllerRef.current) controllerRef.current.abort()
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    smoothRef.current = []
    clearOverlay()
    setRunning(false)
    setFaces([])
  }, [])

  const start = useCallback(async () => {
    setError('')
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('This browser cannot open the camera here. Phones only allow it on HTTPS pages; use the Photo tab, or open the app through an HTTPS link.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingRef.current, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      })
      streamRef.current = stream
      const video = videoRef.current
      video.srcObject = stream
      video.onloadedmetadata = () => { if (video.videoWidth) setAspect(video.videoWidth / video.videoHeight) }
      if (video.play) await video.play()
      runIdRef.current += 1
      runningRef.current = true
      setRunning(true)
      loop(runIdRef.current)
    } catch (e) {
      setError(describeCameraError(e))
    }
  }, [loop])

  const flip = async () => {
    const next = facingRef.current === 'user' ? 'environment' : 'user'
    facingRef.current = next
    setFacing(next)
    if (runningRef.current) {
      stop()
      await start()
    }
  }

  useEffect(() => stop, [stop])

  const main = faces[0]
  const mirrored = facing === 'user'
  return (
    <div className="space-y-4">
      <ErrorBanner message={error} />
      {insecure && !running && (
        <Notice tone="warn">This page is not secure (HTTPS), so most phones will block the live camera. Open the app through an HTTPS link, or use the Photo tab.</Notice>
      )}
      <div className="grid gap-4 md:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div>
          <div className="relative w-full overflow-hidden rounded-2xl border border-line bg-black" style={{ aspectRatio: String(aspect) }}>
            <video ref={videoRef} muted playsInline className="absolute inset-0 h-full w-full object-fill"
                   style={{ transform: mirrored ? 'scaleX(-1)' : 'none' }} aria-label="Camera preview" />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />
            {!running && <div className="absolute inset-0 grid place-items-center text-slate-400">Camera is off</div>}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            {!running ? <Button onClick={start}>Start camera</Button> : <Button variant="ghost" onClick={stop}>Stop</Button>}
            <Button variant="ghost" onClick={flip} aria-label="Switch camera">Switch camera</Button>
            {running && (
              <span className="text-xs text-slate-400" aria-label="Performance">
                {stats.fps ? `${stats.fps.toFixed(1)} updates/s · ` : ''}{stats.ms ? `${Math.round(stats.ms)} ms round trip` : 'starting…'}
              </span>
            )}
          </div>
        </div>
        <Card title="Live emotion">
          {!running && <p className="text-sm text-slate-400">Start the camera to see the emotion of the largest face update live. Nothing is recorded or stored on the server.</p>}
          {running && !main && <p className="text-sm text-slate-300">Looking for a face… face the camera in good light.</p>}
          {main && (
            <div className="space-y-3">
              <BigEmotion label={main.label} confidence={main.confidence} />
              <ProbBars labels={FACE_LABELS} probs={main.probs} />
              {faces.length > 1 && (
                <p className="text-xs text-slate-400">
                  Other faces: {faces.slice(1).map((f, i) => `#${i + 2} ${f.label} ${pct(f.confidence)}`).join(' · ')}
                </p>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
