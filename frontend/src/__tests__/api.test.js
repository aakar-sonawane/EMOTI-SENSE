import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getJSON, postForm, postJSON } from '../api'
import { ok, fail } from './helpers'

beforeEach(() => { globalThis.fetch = vi.fn() })

describe('api client', () => {
  it('returns parsed JSON', async () => {
    fetch.mockResolvedValue(ok({ a: 1 }))
    expect(await getJSON('/api/health')).toEqual({ a: 1 })
  })
  it('sends JSON bodies with the right header', async () => {
    fetch.mockResolvedValue(ok({}))
    await postJSON('/api/text', { text: 'hi' })
    expect(fetch).toHaveBeenCalledWith('/api/text', expect.objectContaining({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"text":"hi"}' }))
  })
  it('uses the FastAPI detail string as the error message', async () => {
    fetch.mockResolvedValue(fail('Text checkpoint not found. Expected: /x/y.pt', 503))
    await expect(postJSON('/api/text', { text: 'a' })).rejects.toMatchObject({ name: 'ApiError', status: 503, message: 'Text checkpoint not found. Expected: /x/y.pt' })
  })
  it('joins validation error arrays', async () => {
    fetch.mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: [{ msg: 'Field required' }, { msg: 'Bad value' }] }) })
    await expect(postForm('/api/face', new FormData())).rejects.toThrow('Field required; Bad value')
  })
  it('falls back to a generic message when the body is not JSON', async () => {
    fetch.mockResolvedValue({ ok: false, status: 502, json: async () => { throw new Error('not json') } })
    await expect(getJSON('/x')).rejects.toThrow('The server returned an error (502).')
  })
  it('turns a network failure into a helpful message', async () => {
    fetch.mockRejectedValue(new TypeError('Failed to fetch'))
    const err = await getJSON('/x').catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.message).toMatch(/Cannot reach the server/)
  })
  it('lets aborts through untouched', async () => {
    const abort = Object.assign(new Error('aborted'), { name: 'AbortError' })
    fetch.mockRejectedValue(abort)
    await expect(getJSON('/x')).rejects.toBe(abort)
  })
})

import { streamChat } from '../api'
import { ndjsonResponse } from './helpers'

describe('streamChat', () => {
  it('parses events that are split across network chunks', async () => {
    const events = [{ type: 'sentiment', label: 'Positive' }, { type: 'delta', text: 'Hi — ना' }, { type: 'done' }]
    fetch.mockResolvedValue(ndjsonResponse(events, { chunk: 5 }))
    const got = []
    await streamChat({ model: 'm', messages: [] }, (e) => got.push(e))
    expect(got).toEqual(events)
    expect(fetch).toHaveBeenCalledWith('/api/assistant/chat', expect.objectContaining({ method: 'POST', body: '{"model":"m","messages":[]}' }))
  })
  it('handles a final line without a trailing newline', async () => {
    const enc = new TextEncoder().encode('{"type":"delta","text":"a"}\n{"type":"done"}')
    let sent = false
    fetch.mockResolvedValue({ ok: true, status: 200, body: { getReader: () => ({ read: async () => (sent ? { done: true } : ((sent = true), { value: enc, done: false })) }) } })
    const got = []
    await streamChat({}, (e) => got.push(e.type))
    expect(got).toEqual(['delta', 'done'])
  })
  it('turns an HTTP error into an ApiError with the server message', async () => {
    fetch.mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: 'The last message must be from the user.' }) })
    await expect(streamChat({}, () => {})).rejects.toMatchObject({ name: 'ApiError', status: 422, message: 'The last message must be from the user.' })
  })
  it('reports an unreachable server and passes aborts through', async () => {
    fetch.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(streamChat({}, () => {})).rejects.toThrow(/Cannot reach the server/)
    const abort = Object.assign(new Error('aborted'), { name: 'AbortError' })
    fetch.mockRejectedValue(abort)
    await expect(streamChat({}, () => {})).rejects.toBe(abort)
  })
})
