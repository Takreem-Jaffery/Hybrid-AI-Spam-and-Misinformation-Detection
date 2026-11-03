import booleanPointInPolygon from '@turf/boolean-point-in-polygon'
import { feature } from 'topojson-client'
import worldTopology from 'world-atlas/countries-110m.json' with { type: 'json' }

const SUPPORTED_COUNTRIES_BY_ID = new Map([
  [36, 'AU'],
  [124, 'CA'],
  [276, 'DE'],
  [356, 'IN'],
  [360, 'ID'],
  [392, 'JP'],
  [410, 'KR'],
  [586, 'PK'],
  [792, 'TR'],
  [826, 'GB'],
  [840, 'US'],
])

const countryFeatures = feature(
  worldTopology,
  worldTopology.objects.countries,
).features

export function getSupportedCountryCode(latitude, longitude) {
  if (
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude) ||
    latitude < -90 ||
    latitude > 90 ||
    longitude < -180 ||
    longitude > 180
  ) {
    return null
  }

  const country = countryFeatures.find((item) =>
    booleanPointInPolygon([longitude, latitude], item),
  )

  return country ? SUPPORTED_COUNTRIES_BY_ID.get(Number(country.id)) ?? null : null
}
