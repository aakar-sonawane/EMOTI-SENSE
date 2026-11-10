import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { HistoryProvider, historyToCsv, useHistory } from '../history'

const wrapper = ({ children }) => <HistoryProvider>{children}</HistoryProvider>
beforeEach(() => localStorage.clear())

describe('history store', () => {
  it('adds entries with a timestamp and persists them', () => {
    const { result } = renderHook(() => useHistory(), { wrapper })
    act(() => result.current.addHistory({ source: 'Text', label: 'Positive', confidence: 0.9, valence: 0.8 }))
    expect(result.current.items).toHaveLength(1)
    expect(result.current.items[0]).toMatchObject({ source: 'Text', label: 'Positive', detail: '' })
    expect(typeof result.current.items[0].ts).toBe('number')
    expect(JSON.parse(localStorage.getItem('emotisense.history.v1'))).toHaveLength(1)
  })
  it('loads previous entries and survives corrupt storage', () => {
    localStorage.setItem('emotisense.history.v1', JSON.stringify([{ ts: 1, source: 'Photo', label: 'Sad' }]))
    expect(renderHook(() => useHistory(), { wrapper }).result.current.items).toHaveLength(1)
    localStorage.setItem('emotisense.history.v1', '{not json')
    expect(renderHook(() => useHistory(), { wrapper }).result.current.items).toEqual([])
  })
  it('keeps at most 500 entries and can be cleared', () => {
    const { result } = renderHook(() => useHistory(), { wrapper })
    act(() => { for (let i = 0; i < 520; i += 1) result.current.addHistory({ source: 'Live camera', label: 'Happy', detail: String(i) }) })
    expect(result.current.items).toHaveLength(500)
    expect(result.current.items[0].detail).toBe('20')
    act(() => result.current.clearHistory())
    expect(result.current.items).toEqual([])
  })
  it('throws a clear error outside the provider', () => {
    expect(() => renderHook(() => useHistory())).toThrow(/HistoryProvider/)
  })
})

describe('CSV export', () => {
  it('escapes commas, quotes and newlines and formats numbers', () => {
    const csv = historyToCsv([{ ts: Date.UTC(2026, 0, 2, 3, 4, 5), source: 'Text', label: 'Positive', confidence: 0.8768, valence: 0.5, detail: 'He said "hi", then\nleft' }])
    const [head, ...rest] = csv.split(/\n(?=\d{4}-)/)
    expect(head).toBe('time,source,label,confidence_percent,valence,detail')
    expect(rest[0]).toBe('2026-01-02T03:04:05.000Z,Text,Positive,87.7,0.500,"He said ""hi"", then\nleft"')
  })
  it('leaves missing numbers empty', () => {
    expect(historyToCsv([{ ts: 0, source: 'Clip', label: 'Neutral', confidence: null, valence: null }]).split('\n')[1]).toBe('1970-01-01T00:00:00.000Z,Clip,Neutral,,,')
  })
})
