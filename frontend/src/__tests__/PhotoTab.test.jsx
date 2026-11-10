import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PhotoTab from '../PhotoTab'
import { fail, installBrowserStubs, ok, renderWithHistory } from './helpers'

const RESULT = {
  label: 'Happy', confidence: 0.82, labels: ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
  probs: [0.01, 0.01, 0.02, 0.82, 0.1, 0.02, 0.02], valence: 0.8, note: null, face: 'data:image/jpeg;base64,AAAA', ms: 250,
}
const pick = (testId, file) => fireEvent.change(screen.getByTestId(testId), { target: { files: [file] } })
const photo = () => new File(['x'], 'me.jpg', { type: 'image/jpeg' })

beforeEach(() => { globalThis.fetch = vi.fn(); installBrowserStubs(); localStorage.clear() })

describe('Photo tab', () => {
  it('uploads the photo with crop=true by default and shows the emotion', async () => {
    fetch.mockResolvedValue(ok(RESULT))
    renderWithHistory(<PhotoTab />)
    pick('photo-camera', photo())
    expect(await screen.findByText('Happy', { selector: 'div' })).toBeInTheDocument()
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('/api/face')
    expect(options.body.get('crop')).toBe('true')
    expect(options.body.get('file').name).toBe('me.jpg')
    expect(screen.getByAltText('Face used by the model')).toHaveAttribute('src', RESULT.face)
    expect(screen.getByRole('list', { name: 'Class probabilities' }).querySelectorAll('li')).toHaveLength(7)
    await waitFor(() => expect(JSON.parse(localStorage.getItem('emotisense.history.v1'))[0]).toMatchObject({ source: 'Photo', label: 'Happy' }))
  })

  it('the phone camera button asks the phone to open the front camera', () => {
    renderWithHistory(<PhotoTab />)
    expect(screen.getByTestId('photo-camera')).toHaveAttribute('capture', 'user')
    expect(screen.getByTestId('photo-gallery')).not.toHaveAttribute('capture')
  })

  it('respects the crop checkbox', async () => {
    fetch.mockResolvedValue(ok(RESULT))
    renderWithHistory(<PhotoTab />)
    await userEvent.click(screen.getByLabelText('Detect and crop the face first'))
    pick('photo-gallery', photo())
    await screen.findByText('Happy', { selector: 'div' })
    expect(fetch.mock.calls[0][1].body.get('crop')).toBe('false')
  })

  it('shows the server error and no result', async () => {
    fetch.mockResolvedValue(fail('Your OpenCV (5.0.0) has no CascadeClassifier because OpenCV 5 removed it.', 503))
    renderWithHistory(<PhotoTab />)
    pick('photo-gallery', photo())
    expect(await screen.findByRole('alert')).toHaveTextContent('OpenCV 5 removed it')
    expect(screen.queryByText('Result')).not.toBeInTheDocument()
  })

  it('warns about low confidence and shows the no-face note', async () => {
    fetch.mockResolvedValue(ok({ ...RESULT, label: 'Neutral', confidence: 0.4, probs: [0.1, 0.1, 0.1, 0.2, 0.4, 0.05, 0.05], note: 'No face was detected, so the whole photo was used.' }))
    renderWithHistory(<PhotoTab />)
    pick('photo-gallery', photo())
    expect(await screen.findByText(/Low confidence/)).toBeInTheDocument()
    expect(screen.getByText(/No face was detected/)).toBeInTheDocument()
  })

  it('requests the heatmap for the same file and shows it', async () => {
    fetch.mockResolvedValueOnce(ok(RESULT)).mockResolvedValueOnce(ok({ label: 'Happy', confidence: 0.82, overlay: 'data:image/png;base64,BBBB' }))
    renderWithHistory(<PhotoTab />)
    pick('photo-gallery', photo())
    await userEvent.click(await screen.findByRole('button', { name: 'Where did the model look?' }))
    expect(await screen.findByAltText('Occlusion heatmap')).toHaveAttribute('src', 'data:image/png;base64,BBBB')
    expect(fetch.mock.calls[1][0]).toBe('/api/face/heatmap')
    expect(fetch.mock.calls[1][1].body.get('file').name).toBe('me.jpg')
  })
})
