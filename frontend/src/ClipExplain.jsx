import { Card, LineChart } from './ui'

const sign = (x) => (x >= 0 ? '+' : '−')
const fmt = (x) => `${sign(x)}${Math.abs(x).toFixed(2)}`

export function ScoreBar({ label, score, strong = false }) {
  const width = (Math.abs(score) / 6) * 100
  const positive = score >= 0
  return (
    <div className="text-sm">
      <div className="mb-1 flex justify-between"><span className={strong ? 'font-semibold text-white' : 'text-slate-300'}>{label}</span><span className="tabular-nums text-slate-300">{fmt(score)}</span></div>
      <div className="relative h-3 rounded-full bg-white/10" role="img" aria-label={`${label}: ${fmt(score)} on a scale from -3 to +3`}>
        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-500" />
        <div data-testid="score-fill" className="absolute inset-y-0 rounded-full"
             style={{ width: `${width}%`, [positive ? 'left' : 'right']: '50%', background: positive ? '#3fb950' : '#e5534b' }} />
      </div>
    </div>
  )
}

export function EffectBars({ title, windows }) {
  const max = Math.max(1e-9, ...windows.map((w) => Math.abs(w.effect)))
  const top = windows.reduce((a, w) => (Math.abs(w.effect) > Math.abs(a.effect) ? w : a), windows[0])
  return (
    <div>
      <h4 className="mb-1 text-sm font-semibold text-slate-200">{title}</h4>
      <div className="flex h-24 items-center gap-1" role="img" aria-label={title}>
        {windows.map((w, i) => (
          <div key={i} className="relative flex h-full flex-1 flex-col justify-center" title={`${w.start_s.toFixed(1)}–${w.end_s.toFixed(1)} s: ${fmt(w.effect)}`}>
            <div className="absolute inset-x-0 top-1/2 h-px bg-slate-600" />
            <div data-testid="effect-bar" className="mx-auto w-full rounded-sm"
                 style={{ height: `${(Math.abs(w.effect) / max) * 46}%`, background: w.effect >= 0 ? '#3fb950' : '#e5534b',
                          alignSelf: 'center', transform: `translateY(${w.effect >= 0 ? '-50%' : '50%'})`, minHeight: 2 }} />
          </div>
        ))}
      </div>
      <div className="flex justify-between text-[10px] text-slate-500"><span>{windows[0].start_s.toFixed(1)} s</span><span>{windows[windows.length - 1].end_s.toFixed(1)} s</span></div>
      <p className="mt-1 text-xs text-slate-400">
        Strongest moment: {top.start_s.toFixed(1)}–{top.end_s.toFixed(1)} s, it {top.effect >= 0 ? 'pulled the score up' : 'pulled the score down'} by {Math.abs(top.effect).toFixed(2)}.
      </p>
    </div>
  )
}

export default function ClipExplain({ data }) {
  const ex = data.explain
  const both = ex.modalities.both
  return (
    <div className="space-y-4">
      <Card title="How audio and video work together">
        <div className="space-y-3">
          <ScoreBar label="Both together (the real prediction)" score={both} strong />
          <ScoreBar label="Audio stream only (video removed)" score={ex.modalities.audio_only} />
          <ScoreBar label="Video stream only (audio removed)" score={ex.modalities.video_only} />
        </div>
        <p className="mt-3 text-sm text-slate-300">
          The fusion gate weighted audio {Math.round(data.gate_audio * 100)}% and video {Math.round(data.gate_video * 100)}% for this clip.
        </p>
        <p className="mt-1 text-xs text-slate-400">
          The model was trained on both streams together, so the single-stream scores show how much the answer depends on each stream; they are not stand-alone predictions.
          Gate weights show what the gate used, not which stream was more informative.
        </p>
      </Card>

      <Card title="Which moments mattered">
        <div className="grid gap-5 md:grid-cols-2">
          <EffectBars title="Voice (audio features)" windows={ex.audio_windows} />
          <EffectBars title="Face (video frames)" windows={ex.video_windows} />
        </div>
        <p className="mt-3 text-xs text-slate-400">
          Each bar hides a short stretch ({ex.window_frames} frames) and measures how the score changes. Green: that moment pulled the sentiment up. Red: it pulled it down.
        </p>
      </Card>

      <Card title="Frames the model looked at">
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
          {ex.frames.map((f, i) => (
            <figure key={i} className="text-center">
              {f.thumb
                ? <img src={f.thumb} alt={`Face at ${f.t.toFixed(1)} seconds`} className="aspect-square w-full rounded-lg object-cover" />
                : <div className="grid aspect-square w-full place-items-center rounded-lg border border-dashed border-line text-[10px] text-slate-500">no face</div>}
              <figcaption className="mt-0.5 text-[10px] text-slate-400">{f.t.toFixed(1)} s</figcaption>
            </figure>
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-400">Frames without a detected face are fed to the model as blank (zeros). Found in {Math.round(data.face_rate * 100)}% of the sampled frames.</p>
      </Card>

      <Card title="Voice features">
        <ul className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          {Object.entries(ex.voice.stats).map(([k, v]) => (
            <li key={k} className="flex justify-between border-b border-line/50 py-1"><span className="text-slate-300">{k}</span><span className="tabular-nums">{v == null ? 'n/a' : v}</span></li>
          ))}
        </ul>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div><p className="mb-1 text-xs text-slate-400">Energy over time</p><LineChart values={ex.voice.rms} color="#4b8bdb" label="Voice energy over time" /></div>
          <div><p className="mb-1 text-xs text-slate-400">Pitch (voiced parts)</p><LineChart values={ex.voice.f0} color="#e3b341" label="Pitch over time" /></div>
        </div>
        <p className="mt-2 text-xs text-slate-400">These are descriptive acoustic statistics, not an emotion prediction. The model itself reads only the first {data.audio_used_s.toFixed(1)} s of audio.</p>
      </Card>
    </div>
  )
}
