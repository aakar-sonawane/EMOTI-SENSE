import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import { HistoryProvider } from '../history'
import { ok } from './helpers'

beforeEach(() => { globalThis.fetch = vi.fn(); localStorage.clear() })
const renderApp = () => render(<HistoryProvider><App /></HistoryProvider>)

describe('App shell', () => {
  it('opens on the Live tab and switches between all tabs', async () => {
    fetch.mockResolvedValue(ok({ models: {}, face_detector: { ok: true, detail: '' }, packages: {} }))
    renderApp()
    expect(screen.getByRole('button', { name: /Start camera/ })).toBeInTheDocument()
    const expected = { Photo: /Take photo/, Text: /Analyse sentiment/, Clip: /Record clip/, History: /Session history/, System: /Self-test/ }
    for (const [tab, marker] of Object.entries(expected)) {
      await userEvent.click(screen.getByRole('button', { name: new RegExp(tab) }))
      expect(await screen.findByText(marker)).toBeInTheDocument()
    }
    expect(screen.getByRole('button', { name: /System/ })).toHaveAttribute('aria-current', 'page')
  })

  it('shows the honesty footer', () => {
    renderApp()
    expect(screen.getByText(/do not reliably reveal how a person feels/)).toBeInTheDocument()
  })

  it('a text prediction appears in History and can be cleared', async () => {
    fetch.mockResolvedValue(ok({ label: 'Positive', confidence: 0.9, labels: ['Negative', 'Neutral', 'Positive'], probs: [0.05, 0.05, 0.9], valence: 0.85, ms: 10 }))
    renderApp()
    await userEvent.click(screen.getByRole('button', { name: /Text/ }))
    await userEvent.type(screen.getByLabelText('Text to analyse'), 'great')
    await userEvent.click(screen.getByRole('button', { name: 'Analyse sentiment' }))
    await screen.findByText('confidence 90%')
    await userEvent.click(screen.getByRole('button', { name: /History/ }))
    expect(await screen.findByText(/1 predictions/)).toBeInTheDocument()
    expect(screen.getByText(/Text · great/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Clear history' }))
    expect(screen.getByText(/Nothing yet/)).toBeInTheDocument()
    await waitFor(() => expect(JSON.parse(localStorage.getItem('emotisense.history.v1'))).toEqual([]))
  })

  it('downloads the history as CSV', async () => {
    globalThis.URL.createObjectURL = vi.fn(() => 'blob:csv')
    globalThis.URL.revokeObjectURL = vi.fn()
    const clicks = []
    const original = HTMLAnchorElement.prototype.click
    HTMLAnchorElement.prototype.click = function () { clicks.push(this.download) }
    localStorage.setItem('emotisense.history.v1', JSON.stringify([{ ts: 1700000000000, source: 'Photo', label: 'Sad', confidence: 0.7, valence: -0.5, detail: 'a.jpg' }]))
    renderApp()
    await userEvent.click(screen.getByRole('button', { name: /History/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Download CSV' }))
    HTMLAnchorElement.prototype.click = original
    expect(clicks).toHaveLength(1)
    expect(clicks[0]).toMatch(/^emotisense_history_.*\.csv$/)
    const blob = URL.createObjectURL.mock.calls[0][0]
    const text = await new Promise((resolve) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.readAsText(blob) })
    expect(text).toContain('Photo,Sad,70.0,-0.500,a.jpg')
  })
})
