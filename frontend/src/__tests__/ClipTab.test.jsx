import { fireEvent, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ClipTab, { ScoreGauge } from '../ClipTab'
import { fail, ok, renderWithHistory } from './helpers'

const RESULT = {
  score: 1.4, raw: 1.4, label: 'Positive', valence: 0.47, gate_audio: 0.35, gate_video: 0.65, face_rate: 0.95, n_frames: 30,
  audio_total_s: 3.0, audio_used_s: 1.6, video_total_s: 3.0, video_used_s: 3.0, ms: { features: 900, model: 20, total: 920 },
}
const pick = (file) => fireEvent.change(screen.getByTestId('clip-gallery'), { target: { files: [file] } })
const clip = () => new File([new Uint8Array(2 * 1048576)], 'talk.mp4', { type: 'video/mp4' })

beforeEach(() => { globalThis.fetch = vi.fn(); localStorage.clear() })

describe('Clip tab', () => {
  it('shows the chosen file and only then offers Analyse', async () => {
    renderWithHistory(<ClipTab />)
    expect(screen.queryByRole('button', { name: 'Analyse clip' })).not.toBeInTheDocument()
    pick(clip())
    expect(screen.getByText(/talk\.mp4 · 2\.0 MB/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Analyse clip' })).toBeEnabled()
  })

  it('uploads the clip and shows score, gate weights and coverage', async () => {
    fetch.mockResolvedValue(ok(RESULT))
    renderWithHistory(<ClipTab />)
    pick(clip())
    await userEvent.click(screen.getByRole('button', { name: 'Analyse clip' }))
    expect(await screen.findByText('Positive')).toBeInTheDocument()
    expect(screen.getByText(/score \+1\.40/)).toBeInTheDocument()
    expect(screen.getByText(/audio 35% · video 65%/)).toBeInTheDocument()
    expect(screen.getByText(/first 1\.6 s of audio \(of 3\.0 s\)/)).toBeInTheDocument()
    expect(fetch.mock.calls[0][0]).toBe('/api/av')
    expect(fetch.mock.calls[0][1].body.get('file').name).toBe('talk.mp4')
    expect(JSON.parse(localStorage.getItem('emotisense.history.v1'))[0]).toMatchObject({ source: 'Audio+Video clip', label: 'Positive' })
  })

  it('warns when a face was rarely found', async () => {
    fetch.mockResolvedValue(ok({ ...RESULT, face_rate: 0.2 }))
    renderWithHistory(<ClipTab />)
    pick(clip())
    await userEvent.click(screen.getByRole('button', { name: 'Analyse clip' }))
    expect(await screen.findByText(/found in only 20% of the sampled frames/)).toBeInTheDocument()
  })

  it('shows a clear error for a clip without audio', async () => {
    fetch.mockResolvedValue(fail('No audio could be extracted. Does the clip have an audio track?', 422))
    renderWithHistory(<ClipTab />)
    pick(clip())
    await userEvent.click(screen.getByRole('button', { name: 'Analyse clip' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('No audio could be extracted')
  })

  it('places the gauge marker proportionally', () => {
    const { rerender, container } = renderWithHistory(<ScoreGauge score={-3} />)
    const marker = () => container.querySelector('.absolute').style.left
    expect(marker()).toBe('0%')
    rerender(<ScoreGauge score={0} />)
    expect(marker()).toBe('50%')
    rerender(<ScoreGauge score={3} />)
    expect(marker()).toBe('100%')
  })
})
