import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TextTab from '../TextTab'
import { fail, ok, renderWithHistory } from './helpers'

const POSITIVE = { label: 'Positive', confidence: 0.91, labels: ['Negative', 'Neutral', 'Positive'], probs: [0.03, 0.06, 0.91], valence: 0.88, ms: 120 }
beforeEach(() => { globalThis.fetch = vi.fn(); localStorage.clear() })

describe('Text tab', () => {
  it('keeps Analyse disabled until there is text', async () => {
    renderWithHistory(<TextTab />)
    expect(screen.getByRole('button', { name: 'Analyse sentiment' })).toBeDisabled()
    await userEvent.type(screen.getByLabelText('Text to analyse'), 'hello')
    expect(screen.getByRole('button', { name: 'Analyse sentiment' })).toBeEnabled()
  })

  it('shows label, confidence, bars and records history', async () => {
    fetch.mockResolvedValue(ok(POSITIVE))
    renderWithHistory(<TextTab />)
    await userEvent.type(screen.getByLabelText('Text to analyse'), 'I love it')
    await userEvent.click(screen.getByRole('button', { name: 'Analyse sentiment' }))
    expect(await screen.findByText('Positive', { selector: 'div' })).toBeInTheDocument()
    expect(screen.getByText('confidence 91%')).toBeInTheDocument()
    expect(within_list('Class probabilities')).toHaveLength(3)
    expect(fetch).toHaveBeenCalledWith('/api/text', expect.objectContaining({ body: JSON.stringify({ text: 'I love it' }) }))
    await waitFor(() => expect(JSON.parse(localStorage.getItem('emotisense.history.v1'))[0]).toMatchObject({ source: 'Text', label: 'Positive', detail: 'I love it' }))
  })

  it('warns on low confidence', async () => {
    fetch.mockResolvedValue(ok({ ...POSITIVE, label: 'Neutral', confidence: 0.4, probs: [0.3, 0.4, 0.3] }))
    renderWithHistory(<TextTab />)
    await userEvent.type(screen.getByLabelText('Text to analyse'), 'meh')
    await userEvent.click(screen.getByRole('button', { name: 'Analyse sentiment' }))
    expect(await screen.findByText(/Low confidence/)).toBeInTheDocument()
  })

  it('shows the server message when the model is missing', async () => {
    fetch.mockResolvedValue(fail('Text checkpoint not found. Expected: /p/TEXT/best_model_finetuned.pt', 503))
    renderWithHistory(<TextTab />)
    await userEvent.type(screen.getByLabelText('Text to analyse'), 'hello')
    await userEvent.click(screen.getByRole('button', { name: 'Analyse sentiment' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Text checkpoint not found')
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument()
  })

  it('explains the prediction word by word (green supports, red opposes)', async () => {
    fetch.mockResolvedValueOnce(ok(POSITIVE)).mockResolvedValueOnce(ok({ words: ['I', 'love', 'it', 'but', 'late'], scores: [0, 0.5, 0.02, -0.01, -0.2], label: 'Positive', confidence: 0.91, truncated: false }))
    renderWithHistory(<TextTab />)
    await userEvent.type(screen.getByLabelText('Text to analyse'), 'I love it but late')
    await userEvent.click(screen.getByRole('button', { name: 'Analyse sentiment' }))
    await userEvent.click(await screen.findByRole('button', { name: 'Explain this prediction' }))
    const words = await screen.findAllByTestId('word')
    expect(words.map((w) => w.textContent)).toEqual(['I', 'love', 'it', 'but', 'late'])
    expect(words[1].style.background).toContain('63, 185, 80')
    expect(words[4].style.background).toContain('229, 83, 75')
    expect(fetch).toHaveBeenLastCalledWith('/api/text/explain', expect.anything())
  })

  it('fills the box from an example chip', async () => {
    renderWithHistory(<TextTab />)
    await userEvent.click(screen.getByRole('button', { name: /The staff were friendly/ }))
    expect(screen.getByLabelText('Text to analyse')).toHaveValue('The staff were friendly and the food was excellent.')
  })
})

function within_list(label) {
  return screen.getByRole('list', { name: label }).querySelectorAll('li')
}
