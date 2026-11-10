import { act, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LiveCamera, { describeCameraError } from '../LiveCamera'
import { later, ok, renderWithHistory } from './helpers'

const LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
const happy = [0.01, 0.01, 0.02, 0.86, 0.06, 0.02, 0.02]
const sad = [0.02, 0.02, 0.05, 0.05, 0.1, 0.7, 0.06]
const face = (probs, box = [0.1, 0.2, 0.3, 0.3]) => ({ box, labels: LABELS, probs, label: LABELS[probs.indexOf(Math.max(...probs))], confidence: Math.max(...probs), valence: 0 })
const frame = (faces) => ({ width: 320, height: 240, faces, ms: { detect: 5, predict: 100, total: 110 } })

let ctx, track, stream, getUserMedia

beforeEach(() => {
  localStorage.clear()
  track = { stop: vi.fn() }
  stream = { getTracks: () => [track] }
  getUserMedia = vi.fn().mockResolvedValue(stream)
  Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia }, configurable: true })
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { get: () => 640, configurable: true })
  Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { get: () => 480, configurable: true })
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { get: () => 400, configurable: true })
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', { get: () => 300, configurable: true })
  ctx = { clearRect: vi.fn(), strokeRect: vi.fn(), fillRect: vi.fn(), fillText: vi.fn(), measureText: () => ({ width: 60 }), drawImage: vi.fn() }
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ctx)
  HTMLCanvasElement.prototype.toBlob = function (cb) { cb(new Blob(['jpg'], { type: 'image/jpeg' })) }
  globalThis.fetch = vi.fn()
})
afterEach(() => { vi.restoreAllMocks() })

const start = () => userEvent.click(screen.getByRole('button', { name: 'Start camera' }))

describe('Live camera', () => {
  it('opens the front camera, uploads small JPEG frames and shows the emotion', async () => {
    fetch.mockImplementation(() => later(ok(frame([face(happy)]))))
    renderWithHistory(<LiveCamera />)
    await start()
    expect(getUserMedia).toHaveBeenCalledWith({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
    expect(await screen.findByText('Happy', { selector: 'div' })).toBeInTheDocument()
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('/api/face/live')
    expect(options.method).toBe('POST')
    expect(options.headers).toEqual({ 'Content-Type': 'image/jpeg' })
    expect(options.body).toBeInstanceOf(Blob)
    const grab = ctx.drawImage.mock.calls[0]
    expect(grab.slice(3)).toEqual([320, 240])                         // downscaled to 320 px wide, aspect kept
    expect(screen.getByRole('list', { name: 'Class probabilities' }).querySelectorAll('li')).toHaveLength(7)
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('mirrors boxes for the front camera and not for the back camera', async () => {
    fetch.mockImplementation(() => later(ok(frame([face(happy, [0.1, 0.2, 0.3, 0.3])]))))
    renderWithHistory(<LiveCamera />)
    await start()
    await waitFor(() => expect(ctx.strokeRect).toHaveBeenCalled())
    expect(ctx.strokeRect.mock.calls[0].map(Math.round)).toEqual([240, 60, 120, 90])   // x = (1 - (0.1 + 0.3)) * 400
    await userEvent.click(screen.getByRole('button', { name: 'Switch camera' }))
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(2))
    expect(getUserMedia.mock.calls[1][0].video.facingMode).toBe('environment')
    ctx.strokeRect.mockClear()                       // only look at boxes drawn after the switch
    await waitFor(() => expect(ctx.strokeRect).toHaveBeenCalled())
    expect(ctx.strokeRect.mock.calls[0].map(Math.round)).toEqual([40, 60, 120, 90])
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('draws the label with the emoji and confidence above the box', async () => {
    fetch.mockImplementation(() => later(ok(frame([face(happy)]))))
    renderWithHistory(<LiveCamera />)
    await start()
    await waitFor(() => expect(ctx.fillText).toHaveBeenCalled())
    expect(ctx.fillText.mock.calls[0][0]).toBe('😄 Happy 86%')
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('smooths the probabilities between frames instead of flipping instantly', async () => {
    let n = 0
    fetch.mockImplementation(() => later(ok(frame([face(n++ === 0 ? happy : sad)]))))
    renderWithHistory(<LiveCamera />)
    await start()
    await screen.findByText('Happy', { selector: 'div' })
    await waitFor(() => expect(n).toBeGreaterThanOrEqual(2))
    // after one 'sad' frame: 0.6*0.86 + 0.4*0.02 = 0.524 (Happy) vs 0.6*0.02 + 0.4*0.7 = 0.292 (Sad) -> still Happy
    expect(screen.getByText('Happy', { selector: 'div' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('Sad', { selector: 'div' })).toBeInTheDocument(), { timeout: 3000 })   // many sad frames later it flips
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('shows a waiting message when no face is found and clears old boxes', async () => {
    fetch.mockImplementation(() => later(ok(frame([]))))
    renderWithHistory(<LiveCamera />)
    await start()
    expect(await screen.findByText(/Looking for a face/)).toBeInTheDocument()
    expect(ctx.strokeRect).not.toHaveBeenCalled()
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('stops the camera tracks and stops sending frames when Stop is pressed', async () => {
    fetch.mockImplementation(() => later(ok(frame([face(happy)]))))
    renderWithHistory(<LiveCamera />)
    await start()
    await screen.findByText('Happy', { selector: 'div' })
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
    expect(track.stop).toHaveBeenCalled()
    const calls = fetch.mock.calls.length
    await act(async () => { await later(null, 80) })
    expect(fetch.mock.calls.length).toBe(calls)
    expect(screen.getByText('Camera is off')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start camera' })).toBeInTheDocument()
  })

  it('sends one frame at a time (no frames pile up while the server is slow)', async () => {
    let inFlight = 0, maxInFlight = 0
    fetch.mockImplementation(() => { inFlight += 1; maxInFlight = Math.max(maxInFlight, inFlight); return later(ok(frame([face(happy)])), 30).then((r) => { inFlight -= 1; return r }) })
    renderWithHistory(<LiveCamera />)
    await start()
    await waitFor(() => expect(fetch.mock.calls.length).toBeGreaterThanOrEqual(3), { timeout: 3000 })
    expect(maxInFlight).toBe(1)
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('shows a helpful message when the server cannot be reached, and recovers', async () => {
    let up = false
    fetch.mockImplementation(() => (up ? later(ok(frame([face(happy)]))) : Promise.reject(new TypeError('Failed to fetch'))))
    renderWithHistory(<LiveCamera />)
    await start()
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the server')
    up = true
    expect(await screen.findByText('Happy', { selector: 'div' }, { timeout: 3000 })).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('writes the dominant emotion to History, but only every few seconds', async () => {
    fetch.mockImplementation(() => later(ok(frame([face(happy)]))))
    renderWithHistory(<LiveCamera />)
    await start()
    await screen.findByText('Happy', { selector: 'div' })
    await act(async () => { await later(null, 150) })
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
    const items = JSON.parse(localStorage.getItem('emotisense.history.v1'))
    expect(items).toHaveLength(1)
    expect(items[0]).toMatchObject({ source: 'Live camera', label: 'Happy' })
  })

  it('lists other faces next to the main one', async () => {
    fetch.mockImplementation(() => later(ok(frame([face(happy), face(sad, [0.6, 0.2, 0.2, 0.2])]))))
    renderWithHistory(<LiveCamera />)
    await start()
    expect(await screen.findByText(/Other faces: #2 Sad 70%/)).toBeInTheDocument()
    await act(async () => { await userEvent.click(screen.getByRole('button', { name: 'Stop' })) })
  })

  it('explains a denied camera permission', async () => {
    getUserMedia.mockRejectedValue(Object.assign(new Error('denied'), { name: 'NotAllowedError' }))
    renderWithHistory(<LiveCamera />)
    await start()
    expect(await screen.findByRole('alert')).toHaveTextContent(/permission was denied/)
    expect(screen.getByRole('button', { name: 'Start camera' })).toBeInTheDocument()
  })

  it('explains the HTTPS requirement when the browser has no camera API', async () => {
    Object.defineProperty(navigator, 'mediaDevices', { value: undefined, configurable: true })
    renderWithHistory(<LiveCamera />)
    await start()
    expect(await screen.findByRole('alert')).toHaveTextContent(/HTTPS/)
  })

  it('releases the camera when the component is removed', async () => {
    fetch.mockImplementation(() => later(ok(frame([face(happy)]))))
    const { unmount } = renderWithHistory(<LiveCamera />)
    await start()
    await screen.findByText('Happy', { selector: 'div' })
    unmount()
    expect(track.stop).toHaveBeenCalled()
  })
})

describe('camera error messages', () => {
  it.each([
    ['NotAllowedError', /permission was denied/],
    ['NotFoundError', /No camera was found/],
    ['NotReadableError', /another app/],
    ['SecurityError', /HTTPS/],
  ])('%s', (name, pattern) => {
    expect(describeCameraError({ name, message: 'x' })).toMatch(pattern)
  })
})
