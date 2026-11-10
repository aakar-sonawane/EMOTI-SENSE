export const FACE_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
export const TEXT_LABELS = ['Negative', 'Neutral', 'Positive']
const POSITIVE = ['Happy', 'Surprise']
const NEGATIVE = ['Angry', 'Disgust', 'Fear', 'Sad']

export const EMOJI = {
  Angry: '😠', Disgust: '🤢', Fear: '😨', Happy: '😄', Neutral: '😐', Sad: '😢', Surprise: '😲',
  Negative: '😞', Positive: '😊',
}

export const COLORS = {
  Angry: '#e5534b', Disgust: '#3fa34d', Fear: '#a371f7', Happy: '#3fb950', Neutral: '#9aa0b4',
  Sad: '#4b8bdb', Surprise: '#e3b341', Negative: '#e5534b', Positive: '#3fb950',
}

/** valence = P(Happy, Surprise) - P(Angry, Disgust, Fear, Sad); same mapping as the server. */
export function faceValence(probs, labels = FACE_LABELS) {
  let v = 0
  labels.forEach((l, i) => {
    if (POSITIVE.includes(l)) v += probs[i]
    if (NEGATIVE.includes(l)) v -= probs[i]
  })
  return v
}

export function argmax(arr) {
  let best = 0
  arr.forEach((v, i) => { if (v > arr[best]) best = i })
  return best
}

export const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`
