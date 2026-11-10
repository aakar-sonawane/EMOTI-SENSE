import { fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ClipTab from '../ClipTab'
import { EffectBars, ScoreBar } from '../ClipExplain'
import { fail, ok, renderWithHistory } from './helpers'
import { render } from '@testing-library/react'

const BASE = {
  score: 1.4, raw: 1.4, label: 'Positive', valence: 0.47, gate_audio: 0.35, gate_video: 0.65, face_rate: 0.75, n_frames: 40,
  audio_total_s: 3.0, audio_used_s: 1.6, video_total_s: 4.0, video_used_s: 4.0, ms: { features: 900, model: 20, total: 920 },
}
const EXPLAINED = {
  ...BASE,
  explain: {
    modalities: { both: 1.4, audio_only: 0.3, video_only: -0.8 },
    audio_windows: [{ start_s: 0, end_s: 0.3, effect: 0.5 }, { start_s: 0.15, end_s: 0.45, effect: -0.2 }, { start_s: 0.3, end_s: 0.6, effect: 0.05 }],
    video_windows: [{ start_s: 0, end_s: 1, effect: -0.4 }, { start_s: 0.5, end_s: 1.5, effect: 0.9 }],
    frames: [{ t: 0, face: true, thumb: 'data:image/jpeg;base64,AAAA' }, { t: 3.5, face: false, thumb: null }],
    voice: { stats: { 'Duration (s)': 3.0, 'Median pitch (Hz)': 182.4, 'Pitch variability (Hz, std)': null }, t: [0, 1, 2, 3], rms: [0.1, 0.4, 0.2, 0.3], f0: [null, 180, 190, null] },
    window_frames: 10,
  },
}
const pick = (file) => fireEvent.change(screen.getByTestId('clip-gallery'), { target: { files: [file] } })
const clip = () => new File(['v'], 'talk.mp4', { type: 'video/mp4' })

beforeEach(() => { globalThis.fetch = vi.fn(); localStorage.clear() })

async function analyseThenExplain() {
  fetch.mockResolvedValueOnce(ok(BASE)).mockResolvedValueOnce(ok(EXPLAINED))
  renderWithHistory(<ClipTab />)
  pick(clip())
  await userEvent.click(screen.getByRole('button', { name: 'Analyse clip' }))
  await userEvent.click(await screen.findByRole('button', { name: 'Why this score?' }))
}

describe('Why this score? (audio + video explanation)', () => {
  it('uploads the same clip to /api/av/explain and shows how both streams contribute', async () => {
    await analyseThenExplain()
    expect(await screen.findByText('How audio and video work together')).toBeInTheDocument()
    expect(fetch.mock.calls[1][0]).toBe('/api/av/explain')
    expect(fetch.mock.calls[1][1].body.get('file').name).toBe('talk.mp4')
    expect(screen.getByRole('img', { name: /Both together \(the real prediction\): \+1\.40/ })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Audio stream only \(video removed\): \+0\.30/ })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Video stream only \(audio removed\): −0\.80/ })).toBeInTheDocument()
    expect(screen.getByText(/weighted audio 35% and video 65%/)).toBeInTheDocument()
    expect(screen.getByText(/not stand-alone predictions/)).toBeInTheDocument()
  })

  it('draws one coloured bar per time window and names the strongest moment', async () => {
    await analyseThenExplain()
    await screen.findByText('Which moments mattered')
    const bars = screen.getAllByTestId('effect-bar')
    expect(bars).toHaveLength(5)
    expect(bars[0].style.background).toMatch(/63, 185, 80/)      // +0.5 -> green
    expect(bars[1].style.background).toMatch(/229, 83, 75/)      // -0.2 -> red
    expect(screen.getByText(/Strongest moment: 0\.0–0\.3 s, it pulled the score up by 0\.50/)).toBeInTheDocument()
    expect(screen.getByText(/Strongest moment: 0\.5–1\.5 s, it pulled the score up by 0\.90/)).toBeInTheDocument()
  })

  it('shows the sampled frames, marks missing faces, and lists voice statistics', async () => {
    await analyseThenExplain()
    expect(await screen.findByAltText('Face at 0.0 seconds')).toHaveAttribute('src', 'data:image/jpeg;base64,AAAA')
    expect(screen.getByText('no face')).toBeInTheDocument()
    expect(screen.getByText(/Found in 75% of the sampled frames/)).toBeInTheDocument()
    const stats = screen.getByText('Median pitch (Hz)').closest('li')
    expect(within(stats).getByText('182.4')).toBeInTheDocument()
    expect(within(screen.getByText('Pitch variability (Hz, std)').closest('li')).getByText('n/a')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Voice energy over time' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Pitch over time' })).toBeInTheDocument()
    expect(screen.getByText(/not an emotion prediction/)).toBeInTheDocument()
  })

  it('hides the button after it ran, and shows server errors', async () => {
    fetch.mockResolvedValueOnce(ok(BASE)).mockResolvedValueOnce(fail('No audio could be extracted.', 422))
    renderWithHistory(<ClipTab />)
    pick(clip())
    await userEvent.click(screen.getByRole('button', { name: 'Analyse clip' }))
    await userEvent.click(await screen.findByRole('button', { name: 'Why this score?' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No audio could be extracted')
    expect(screen.getByRole('button', { name: 'Why this score?' })).toBeEnabled()          // can retry
  })

  it('a new analysis clears the old explanation', async () => {
    await analyseThenExplain()
    await screen.findByText('How audio and video work together')
    fetch.mockResolvedValueOnce(ok(BASE))
    await userEvent.click(screen.getByRole('button', { name: 'Analyse clip' }))
    await screen.findByRole('button', { name: 'Why this score?' })
    expect(screen.queryByText('How audio and video work together')).not.toBeInTheDocument()
  })
})

describe('chart pieces', () => {
  it('ScoreBar fills to the right for positive and to the left for negative scores', () => {
    const { rerender, getByTestId } = render(<ScoreBar label="x" score={3} />)
    expect(getByTestId('score-fill').style.width).toBe('50%')
    expect(getByTestId('score-fill').style.left).toBe('50%')
    rerender(<ScoreBar label="x" score={-1.5} />)
    expect(getByTestId('score-fill').style.width).toBe('25%')
    expect(getByTestId('score-fill').style.right).toBe('50%')
  })
  it('EffectBars handles a single window', () => {
    render(<EffectBars title="t" windows={[{ start_s: 0, end_s: 1, effect: -0.3 }]} />)
    expect(screen.getAllByTestId('effect-bar')).toHaveLength(1)
    expect(screen.getByText(/pulled the score down by 0\.30/)).toBeInTheDocument()
  })
})
