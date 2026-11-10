import { useState } from 'react'
import AssistantTab from './AssistantTab'
import ClipTab from './ClipTab'
import HistoryTab from './HistoryTab'
import LiveCamera from './LiveCamera'
import PhotoTab from './PhotoTab'
import SystemTab from './SystemTab'
import TextTab from './TextTab'

const TABS = [
  { id: 'live', label: 'Live', icon: '🎥', view: LiveCamera },
  { id: 'photo', label: 'Photo', icon: '📷', view: PhotoTab },
  { id: 'text', label: 'Text', icon: '📝', view: TextTab },
  { id: 'clip', label: 'Clip', icon: '🎬', view: ClipTab },
  { id: 'assistant', label: 'Assistant', icon: '💬', view: AssistantTab },
  { id: 'history', label: 'History', icon: '🕘', view: HistoryTab },
  { id: 'system', label: 'System', icon: '⚙️', view: SystemTab },
]

export default function App() {
  const [active, setActive] = useState('live')
  const Current = TABS.find((t) => t.id === active).view
  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-3 pb-28 pt-[max(1rem,env(safe-area-inset-top))] md:pb-8">
      <header className="mb-4 flex flex-wrap items-end justify-between gap-2 px-1">
        <div>
          <h1 className="font-display text-3xl font-bold tracking-tight">🧠 EmotiSense</h1>
          <p className="text-sm text-slate-400">Emotion and sentiment from faces, text and speech</p>
        </div>
        <nav aria-label="Sections" className="fixed inset-x-0 bottom-0 z-20 border-t border-line bg-ink/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:static md:border-0 md:bg-transparent md:pb-0">
          <ul className="mx-auto flex max-w-5xl justify-around md:gap-1">
            {TABS.map((t) => (
              <li key={t.id} className="flex-1 md:flex-none">
                <button type="button" onClick={() => setActive(t.id)} aria-current={active === t.id ? 'page' : undefined}
                        className={`flex min-h-[56px] w-full flex-col items-center justify-center gap-0.5 px-2 text-xs font-semibold md:min-h-[44px] md:flex-row md:gap-2 md:rounded-xl md:px-4 md:text-sm ${
                          active === t.id ? 'text-white md:bg-gradient-to-r md:from-accent md:to-accent2' : 'text-slate-400 hover:text-white'}`}>
                  <span className="text-xl md:text-base" aria-hidden="true">{t.icon}</span>{t.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      <main className="flex-1"><Current /></main>
      <footer className="mt-8 px-1 text-xs text-slate-500">
        Predictions are automatic estimates from machine-learning models. Facial expressions and voice do not reliably reveal how a person feels, so use the results as a hint, not as a fact or a diagnosis.
      </footer>
    </div>
  )
}
