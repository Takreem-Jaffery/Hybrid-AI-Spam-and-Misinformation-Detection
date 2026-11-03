import assert from 'node:assert/strict'
import test from 'node:test'
import { getSupportedCountryCode } from '../src/location.js'

const knownLocations = [
  ['Australia', -33.8688, 151.2093, 'AU'],
  ['Canada', 45.4215, -75.6972, 'CA'],
  ['Germany', 52.52, 13.405, 'DE'],
  ['India', 28.6139, 77.209, 'IN'],
  ['Indonesia', -6.2088, 106.8456, 'ID'],
  ['Japan', 35.6762, 139.6503, 'JP'],
  ['Pakistan', 33.6844, 73.0479, 'PK'],
  ['South Korea', 37.5665, 126.978, 'KR'],
  ['Türkiye', 39.9334, 32.8597, 'TR'],
  ['United Kingdom', 51.5074, -0.1278, 'GB'],
  ['United States', 38.8977, -77.0365, 'US'],
]

for (const [name, latitude, longitude, expected] of knownLocations) {
  test(`maps a location in ${name} to its supported country code`, () => {
    assert.equal(getSupportedCountryCode(latitude, longitude), expected)
  })
}

test('returns null for invalid coordinates and unsupported countries', () => {
  assert.equal(getSupportedCountryCode(Number.NaN, 0), null)
  assert.equal(getSupportedCountryCode(91, 0), null)
  assert.equal(getSupportedCountryCode(0, 0), null)
  assert.equal(getSupportedCountryCode(48.8566, 2.3522), null)
})
