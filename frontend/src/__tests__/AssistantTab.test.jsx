import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AssistantTab from '../AssistantTab'
import { ndjsonResponse, ok, renderWithHistory } from './helpers'

const STATUS = { available: true, url: 'http://localhost:11434', models: ['llama3:8b', 'qwen3:8b'], preferred: 'qwen3:8b', error: null }
const DOWN = { available: false, url: 'http://localhost:11434', models: [], preferred: null, error: 'Cannot reach Ollama at http://localhost:11434 (refused). Start it with: ollama serve' }
const SENTIMENT = { type: 'sentiment', label: 'Negative', confidence: 0.81, valence: -0.7, probs: [0.81, 0.12, 0.07], labels: ['Negative', 'Neutral', 'Positive'] }
const reply = (text, extra = []) => [...extra, ...text.match(/.{1,5}/g).map((t) => ({ type: 'delta', text: t })), { type: 'done' }]

function mockApi({ status = STATUS, chat = () => ndjsonResponse(reply('Hello there')) } = {}) {
  globalThis.fetch = vi.fn((url, options) => {
    if (url === '/api/assistant/status') return Promise.resolve(ok(typeof status === 'function' ? status() : status))
    if (url === '/api/assistant/start') return Promise.resolve(ok({ started: true, phase: 'checking' }))
    if (url === '/api/assistant/chat') return Promise.resolve(chat(options))
    return Promise.reject(new Error(`unexpected ${url}`))
  })
}
const chatBody = (n = 0) => JSON.parse(fetch.mock.calls.filter((c) => c[0] === '/api/assistant/chat')[n][1].body)
const typeAndSend = async (text) => { await userEvent.type(await screen.findByLabelText('Your message'), text); await userEvent.click(screen.getByRole('button', { name: 'Send' })) }

beforeEach(() => { localStorage.clear(); Element.prototype.scrollIntoView = vi.fn() })
afterEach(() => { vi.useRealTimers() })

describe('Assistant tab', () => {
  it('explains how to start Ollama when it is not running, and can check again', async () => {
    mockApi({ status: DOWN })
    renderWithHistory(<AssistantTab />)
    expect(await screen.findByRole('alert')).toHaveTextContent('ollama serve')
    expect(screen.getByText(/ollama pull qwen3:8b/)).toBeInTheDocument()
    mockApi()
    await userEvent.click(screen.getByRole('button', { name: 'Check again' }))
    expect(await screen.findByLabelText('Your message')).toBeInTheDocument()
    expect(fetch).toHaveBeenCalledWith('/api/assistant/start', { method: 'POST' })      // Check again also asks the server to start Ollama
  })

  it('offers the installed models and preselects the Qwen3 one', async () => {
    mockApi()
    renderWithHistory(<AssistantTab />)
    const select = await screen.findByLabelText('Local model')
    expect(within(select).getAllByRole('option').map((o) => o.value)).toEqual(['llama3:8b', 'qwen3:8b'])
    expect(select).toHaveValue('qwen3:8b')
  })

  it('sends the conversation, streams the answer in pieces and shows the sentiment chip', async () => {
    mockApi({ chat: () => ndjsonResponse(reply('That sounds stressful. Want to talk it through?', [SENTIMENT]), { delayMs: 3 }) })
    renderWithHistory(<AssistantTab />)
    await typeAndSend('I am so nervous about my viva')
    expect(await screen.findByText(/That sounds stressful\. Want to talk it through\?/)).toBeInTheDocument()
    expect(screen.getByText(/Negative · 81%/)).toBeInTheDocument()
    expect(screen.getByLabelText('Your message')).toHaveValue('')
    expect(chatBody()).toEqual({ model: 'qwen3:8b', style: 'Supportive', temperature: 0.7, analyse: true, face_emotion: null,
                                 messages: [{ role: 'user', content: 'I am so nervous about my viva' }] })
    await waitFor(() => expect(JSON.parse(localStorage.getItem('emotisense.history.v1'))[0]).toMatchObject({ source: 'Assistant', label: 'Negative' }))
  })

  it('keeps the whole conversation in the next request', async () => {
    mockApi()
    renderWithHistory(<AssistantTab />)
    await typeAndSend('first')
    await screen.findByText('Hello there')
    await typeAndSend('second')
    await waitFor(() => expect(fetch.mock.calls.filter((c) => c[0] === '/api/assistant/chat')).toHaveLength(2))
    expect(chatBody(1).messages).toEqual([{ role: 'user', content: 'first' }, { role: 'assistant', content: 'Hello there' }, { role: 'user', content: 'second' }])
  })

  it('shows the helpline note from the server above the reply', async () => {
    mockApi({ chat: () => ndjsonResponse(reply('I am here with you.', [{ type: 'crisis', note: 'Please call Tele-MANAS on 14416 or 112.' }])) })
    renderWithHistory(<AssistantTab />)
    await typeAndSend('I want to die')
    expect(await screen.findByRole('note')).toHaveTextContent('14416')
    expect(await screen.findByText('I am here with you.')).toBeInTheDocument()
  })

  it('shows an error from Ollama and removes the empty assistant bubble', async () => {
    mockApi({ chat: () => ndjsonResponse([{ type: 'error', message: 'Ollama answered with an error (404): model not found. Install it with: ollama pull qwen3:8b' }]) })
    renderWithHistory(<AssistantTab />)
    await typeAndSend('hello')
    expect(await screen.findByRole('alert')).toHaveTextContent('ollama pull qwen3:8b')
    const bubbles = screen.getAllByRole('listitem')
    expect(bubbles).toHaveLength(1)                       // only the user's message is left
  })

  it('reports an unreachable server', async () => {
    globalThis.fetch = vi.fn((url) => (url === '/api/assistant/status' ? Promise.resolve(ok(STATUS)) : Promise.reject(new TypeError('Failed to fetch'))))
    renderWithHistory(<AssistantTab />)
    await typeAndSend('hello')
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the server')
  })

  it('Stop cancels the answer without showing an error', async () => {
    let signal
    mockApi({ chat: (options) => { signal = options.signal; return ndjsonResponse(reply('a very long answer that keeps going and going'), { chunk: 4, delayMs: 20, signal }) } })
    renderWithHistory(<AssistantTab />)
    await typeAndSend('tell me a story')
    await userEvent.click(await screen.findByRole('button', { name: 'Stop' }))
    expect(signal.aborted).toBe(true)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(await screen.findByRole('button', { name: 'Send' })).toBeInTheDocument()
  })

  it('shares the latest camera emotion only when asked and only if it is recent', async () => {
    const now = Date.now()
    localStorage.setItem('emotisense.history.v1', JSON.stringify([
      { ts: now - 10000, source: 'Live camera', label: 'Sad', confidence: 0.7, valence: -0.5, detail: '' }]))
    mockApi()
    renderWithHistory(<AssistantTab />)
    expect(await screen.findByText(/Right now: .*Sad \(\d+ s ago\)/)).toBeInTheDocument()
    await typeAndSend('hi')
    await screen.findByText('Hello there')
    expect(chatBody(0).face_emotion).toBeNull()           // toggle off by default
    await userEvent.click(screen.getByLabelText(/share my latest face emotion/i))
    await typeAndSend('hi again')
    await waitFor(() => expect(fetch.mock.calls.filter((c) => c[0] === '/api/assistant/chat')).toHaveLength(2))
    expect(chatBody(1).face_emotion).toMatchObject({ label: 'Sad', confidence: 0.7 })
    expect(chatBody(1).face_emotion.age_s).toBeGreaterThanOrEqual(10)
  })

  it('does not offer an old camera emotion', async () => {
    localStorage.setItem('emotisense.history.v1', JSON.stringify([{ ts: Date.now() - 600000, source: 'Photo', label: 'Happy', confidence: 0.9 }]))
    mockApi()
    renderWithHistory(<AssistantTab />)
    expect(await screen.findByText(/No face emotion in the last 2 minutes/)).toBeInTheDocument()
    await userEvent.click(screen.getByLabelText(/share my latest face emotion/i))
    await typeAndSend('hi')
    await screen.findByText('Hello there')
    expect(chatBody().face_emotion).toBeNull()
  })

  it('can switch the sentiment estimate off and change the style', async () => {
    mockApi()
    renderWithHistory(<AssistantTab />)
    await userEvent.click(await screen.findByLabelText(/Estimate how each of my messages feels/))
    await userEvent.selectOptions(screen.getByLabelText('Reply style'), 'Professional')
    await typeAndSend('hello')
    await screen.findByText('Hello there')
    expect(chatBody()).toMatchObject({ analyse: false, style: 'Professional' })
  })

  it('Enter sends, Shift+Enter does not', async () => {
    mockApi()
    renderWithHistory(<AssistantTab />)
    const box = await screen.findByLabelText('Your message')
    await userEvent.type(box, 'line one{Shift>}{Enter}{/Shift}line two')
    expect(fetch.mock.calls.filter((c) => c[0] === '/api/assistant/chat')).toHaveLength(0)
    expect(box).toHaveValue('line one\nline two')
    await userEvent.type(box, '{Enter}')
    await screen.findByText('Hello there')
    expect(chatBody().messages[0].content).toBe('line one\nline two')
  })

  it('disables Send for an empty message and starts over with New conversation', async () => {
    mockApi()
    renderWithHistory(<AssistantTab />)
    expect(await screen.findByRole('button', { name: 'Send' })).toBeDisabled()
    await typeAndSend('hello')
    await screen.findByText('Hello there')
    await userEvent.click(screen.getByRole('button', { name: 'New conversation' }))
    expect(screen.queryByText('Hello there')).not.toBeInTheDocument()
    expect(screen.getByText(/Say something/)).toBeInTheDocument()
  })

  it('summarises the mood after a few messages', async () => {
    let n = 0
    const moods = [-0.8, -0.2, 0.5, 0.7]
    mockApi({ chat: () => ndjsonResponse(reply('ok', [{ ...SENTIMENT, valence: moods[n++] }])) })
    renderWithHistory(<AssistantTab />)
    for (const t of ['one', 'two', 'three', 'four']) { await typeAndSend(t); await waitFor(() => expect(n).toBeGreaterThan(moods.indexOf(moods[n - 1]))); await screen.findByRole('button', { name: 'Send' }) }
    expect(await screen.findByText(/4 messages · average/)).toBeInTheDocument()
    expect(screen.getByText(/from -0\.50 at the start to 0\.60 now/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Mood of your messages' })).toBeInTheDocument()
  })

  it('shows that Ollama is being started for you and moves on by itself when it is ready', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let phase = 'starting'
    mockApi({ status: () => (phase === 'starting'
      ? { ...DOWN, starting: true, warming: false, phase: 'starting', phase_message: 'Starting Ollama…' }
      : { ...STATUS, starting: false, warming: false, phase: 'ready', phase_message: 'Ollama is ready.' }) })
    renderWithHistory(<AssistantTab />)
    expect(await screen.findByText('Starting Ollama…')).toBeInTheDocument()
    expect(screen.getByText(/You do not need to run anything/)).toBeInTheDocument()
    expect(screen.queryByText('ollama serve')).not.toBeInTheDocument()
    phase = 'ready'
    await vi.advanceTimersByTimeAsync(1700)
    expect(await screen.findByLabelText('Your message')).toBeInTheDocument()
  })

  it('tells you while the model is still loading into memory, and removes the note when done', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let warming = true
    mockApi({ status: () => ({ ...STATUS, starting: false, warming, phase: warming ? 'warming' : 'ready', phase_message: warming ? 'Loading qwen3:8b into memory…' : 'Ollama is ready.' }) })
    renderWithHistory(<AssistantTab />)
    expect(await screen.findByText(/Loading qwen3:8b into memory…/)).toBeInTheDocument()
    expect(screen.getByLabelText('Your message')).toBeInTheDocument()          // chat is usable meanwhile
    warming = false
    await vi.advanceTimersByTimeAsync(1700)
    await waitFor(() => expect(screen.queryByText(/Loading qwen3:8b into memory…/)).not.toBeInTheDocument())
  })

  it('shows the reason when Ollama is not installed', async () => {
    mockApi({ status: { ...DOWN, starting: false, phase: 'missing', error: 'Ollama is not installed on this computer (or not on PATH). Install it from https://ollama.com and run: ollama pull qwen3:8b' } })
    renderWithHistory(<AssistantTab />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Install it from https://ollama.com')
  })
})
