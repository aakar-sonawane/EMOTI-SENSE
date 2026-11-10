import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import SystemTab from '../SystemTab'
import { ok, renderWithHistory } from './helpers'

const HEALTH = {
  models: {
    text: { file: '/p/TEXT/best_model_finetuned.pt', found: true, loaded: true },
    face: { file: '/p/FACE/m.keras', found: true, loaded: false },
    av: { file: '/p/AUDIO AND VIDEO/best_model.pt', found: false, loaded: false },
  },
  face_detector: { ok: false, detail: 'OpenCV 5.0.0 has no CascadeClassifier (use opencv-python-headless<5)' },
  packages: { torch: true, tensorflow: false },
}
beforeEach(() => { globalThis.fetch = vi.fn() })

describe('System tab', () => {
  it('shows which model files exist and what is wrong', async () => {
    fetch.mockResolvedValue(ok(HEALTH))
    renderWithHistory(<SystemTab />)
    expect(await screen.findByText('Text model (BERT + RoBERTa)')).toBeInTheDocument()
    expect(screen.getAllByText('file found')).toHaveLength(2)
    expect(screen.getByText('file missing')).toBeInTheDocument()
    expect(screen.getByText('loaded')).toBeInTheDocument()
    expect(screen.getByText('needs fix')).toBeInTheDocument()
    expect(screen.getByText(/OpenCV 5.0.0 has no CascadeClassifier/)).toBeInTheDocument()
    expect(screen.getByText(/torch ✓ · tensorflow ✗/)).toBeInTheDocument()
  })

  it('runs the self-test and lists PASS and FAIL', async () => {
    fetch.mockResolvedValueOnce(ok(HEALTH)).mockResolvedValueOnce(ok({ results: [
      { name: 'Text model: clearly positive sentence', ok: true, detail: 'predicted Positive (97%)', ms: 3 },
      { name: 'Audio + video model: loads', ok: false, detail: 'ModelError: not found', ms: 1 },
    ] }))
    renderWithHistory(<SystemTab />)
    await userEvent.click(await screen.findByRole('button', { name: 'Run self-test' }))
    expect(await screen.findByText('predicted Positive (97%)')).toBeInTheDocument()
    expect(screen.getByText('PASS')).toBeInTheDocument()
    expect(screen.getByText('FAIL')).toBeInTheDocument()
    expect(fetch).toHaveBeenLastCalledWith('/api/selftest', undefined)
  })

  it('shows an error when the server is unreachable', async () => {
    fetch.mockRejectedValue(new TypeError('Failed to fetch'))
    renderWithHistory(<SystemTab />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the server')
  })
})
