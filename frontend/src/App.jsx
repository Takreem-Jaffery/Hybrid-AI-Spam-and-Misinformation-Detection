import { useState, useEffect, useRef } from 'react'
import { predict, getAdvice, reviewArticle } from './api.js'
import {
  COUNTRY_REPORTING_RESOURCES,
  GLOBAL_REPORTING_RESOURCES,
  REPORTING_COUNTRIES,
} from './helplines.js'
import { getSupportedCountryCode } from './location.js'
import { applyTheme } from './theme.js'
import './styles.css'

const MAX_CHARS = 500

function toView(data) {
  const risk = (data.risk || 'Low').toUpperCase()
  let prediction = 'BENIGN'
  if (data.label === 'Spam') prediction = 'SPAM'
  else if (risk === 'MEDIUM') prediction = 'SUSPICIOUS'
  return {
    prediction,
    riskLevel: risk,
    riskColor: data.riskColor || 'var(--risk-low)',
    score: Math.round(data.spam_score),
    reasons: data.reasons?.length ? data.reasons : ['No suspicious patterns detected'],
    modelUsed: data.model_used || '',
  }
}

export default function App() {
  const [theme, setTheme] = useState('light')
  const [mode, setMode] = useState('message')
  const [message, setMessage] = useState('')
  const [articleTitle, setArticleTitle] = useState('')
  const [articleBody, setArticleBody] = useState('')
  const [forwarded, setForwarded] = useState(false)
  const [source, setSource] = useState('unknown')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [advice, setAdvice] = useState(null)
  const [adviceLoading, setAdviceLoading] = useState(false)
  const [adviceError, setAdviceError] = useState(null)
  const [adviceOpen, setAdviceOpen] = useState(false)
  const [showReport, setShowReport] = useState(false)
  const [countryCode, setCountryCode] = useState('')
  const [locationStatus, setLocationStatus] = useState('')
  const [locationLoading, setLocationLoading] = useState(false)
  const [articleResult, setArticleResult] = useState(null)
  const [articleLoading, setArticleLoading] = useState(false)
  const [articleError, setArticleError] = useState(null)
  const resultRef = useRef(null)

  useEffect(() => { applyTheme(theme) }, [theme])

  const selectedCountry = REPORTING_COUNTRIES.find(country => country.code === countryCode)
  const countryResources = countryCode ? COUNTRY_REPORTING_RESOURCES[countryCode] : []

  const clearOutputs = () => {
    setResult(null); setError(null); setAdvice(null); setAdviceError(null)
    setAdviceOpen(false); setShowReport(false)
    setArticleResult(null); setArticleError(null)
  }

  const detectCountry = () => {
    if (!window.isSecureContext) {
      setLocationStatus('Device-location detection requires HTTPS or localhost. Choose a country manually.')
      return
    }

    if (!navigator.geolocation) {
      setLocationStatus('Device location is unavailable in this browser. Choose a country manually.')
      return
    }

    setLocationLoading(true)
    setLocationStatus('')
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        const detectedCode = getSupportedCountryCode(coords.latitude, coords.longitude)
        const detected = REPORTING_COUNTRIES.find(country => country.code === detectedCode)
        setCountryCode(detected?.code ?? '')
        setLocationStatus(detected
          ? `Detected ${detected.name}. You can change the country below.`
          : 'This location is outside the countries currently covered. Choose a country manually.')
        setLocationLoading(false)
      },
      error => {
        const messages = {
          1: 'Location permission was denied. Choose a country manually if you prefer.',
          2: 'The browser could not determine your location. Choose a country manually.',
          3: 'Location detection timed out. Try again or choose a country manually.',
        }
        setLocationStatus(messages[error.code] || 'Location detection failed. Choose a country manually.')
        setLocationLoading(false)
      },
      { enableHighAccuracy: false, maximumAge: 60_000, timeout: 10_000 },
    )
  }

  const analyze = async () => {
    if (!message.trim()) return
    setLoading(true); clearOutputs()
    try {
      setResult(toView(await predict(message, forwarded, source)))
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchAdvice = async () => {
    setAdviceLoading(true); setAdviceError(null); setAdviceOpen(true)
    try {
      setAdvice(await getAdvice(message, forwarded, source))
    } catch (e) {
      setAdviceError(e.message)
    } finally {
      setAdviceLoading(false)
    }
  }

  const analyzeArticle = async () => {
    if (!articleTitle.trim() || !articleBody.trim()) return
    setArticleLoading(true); clearOutputs()
    try {
      setArticleResult(await reviewArticle(articleTitle.trim(), articleBody.trim()))
    } catch (e) {
      setArticleError(e.message)
    } finally {
      setArticleLoading(false)
    }
  }

  const reset = () => {
    setMessage(''); setArticleTitle(''); setArticleBody('')
    setForwarded(false); setSource('unknown'); clearOutputs()
  }

  const title = result?.prediction === 'SPAM'
    ? 'Threat detected'
    : result?.prediction === 'SUSPICIOUS' ? 'Exercise caution' : 'No threat detected'
  const summary = result?.prediction === 'SPAM'
    ? 'This message has patterns commonly found in spam or phishing.'
    : result?.prediction === 'SUSPICIOUS'
      ? 'Some signals deserve a closer look before you respond.'
      : 'No strong spam signals were detected in this message.'

  return (
    <div className="app-wrap">
      <div className="container">
        <header className="header">
          <a className="brand" href="/" aria-label="Message Risk Analyzer home">
            <span className="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 32 32" fill="none">
                <path d="M16 3.5 26 7v7.4c0 6.5-4.3 11.6-10 14.1C10.3 26 6 20.9 6 14.4V7l10-3.5Z" />
                <path d="m11.5 15.7 3 3 6-6.2" />
              </svg>
            </span>
            <span className="brand-copy">
              <span className="brand-name">Message Risk Analyzer</span>
              <span className="brand-caption">MESSAGE &amp; ARTICLE REVIEW</span>
            </span>
          </a>
          <div className="header-actions">
            <div className="local-status"><span className="status-dot" /> LOCAL ANALYSIS</div>
            <button className="theme-toggle" onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))} title="Toggle theme" aria-label="Toggle theme">
              <span aria-hidden="true">{theme === 'dark' ? '☼' : '☾'}</span>
            </button>
          </div>
        </header>

        <main>
          <section className="hero">
            <div className="hero-copy">
              <div className="eyebrow"><span className="eyebrow-line" /> MESSAGE &amp; ARTICLE ANALYSIS</div>
              <h1>Pause. Check.<br /><span>Then respond.</span></h1>
              <p>{mode === 'message'
                ? 'Inspect a suspicious message for common spam and phishing signals before you take action.'
                : 'Review an article’s headline and text against historical veracity labels and clickbait annotations.'}</p>
            </div>
            <div className="hero-aside">
              <div className="hero-aside-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none"><path d="M12 3 20 6v6c0 5-3.4 8.8-8 10-4.6-1.2-8-5-8-10V6l8-3Z" /><path d="m8.5 12 2.2 2.2 4.8-5" /></svg>
              </div>
              <div><strong>Stay one step ahead</strong><span>Clear signals. Safer decisions.</span></div>
            </div>
          </section>

          <div className="mode-switch" role="group" aria-label="Analysis mode">
            <button type="button" aria-pressed={mode === 'message'} className={mode === 'message' ? 'is-active' : ''} onClick={() => { setMode('message'); clearOutputs() }}>Message spam scan</button>
            <button type="button" aria-pressed={mode === 'article'} className={mode === 'article' ? 'is-active' : ''} onClick={() => { setMode('article'); clearOutputs() }}>Article review</button>
          </div>

          <div className="workspace-grid">
            <section className="panel analysis-panel">
              <div className="panel-heading">
                <div>
                  <div className="section-kicker">{mode === 'message' ? 'MESSAGE SCAN' : 'ARTICLE REVIEW'}</div>
                  <h2>{mode === 'message' ? 'Analyze a message' : 'Review an article'}</h2>
                </div>
                <span className="panel-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none"><path d="M4 5.5h16M4 10.5h16M4 15.5h10M4 20.5h7" /><circle cx="18" cy="17" r="3" /><path d="m20.2 19.2 1.8 1.8" /></svg>
                </span>
              </div>
              <p className="panel-description">{mode === 'message'
                ? 'Paste the message text below. You can include links or suspicious wording for a more useful scan.'
                : 'Enter a headline and article body to review corpus-based veracity labels and headline clickbait patterns.'}</p>

              {mode === 'message' ? (
                <>
              <label className="field-label" htmlFor="msg">Message content <span>REQUIRED</span></label>
              <div className="textarea-wrap">
                <textarea
                  id="msg"
                  className="textarea"
                  value={message}
                  onChange={e => setMessage(e.target.value)}
                  placeholder="Paste or type the message you want to check..."
                  maxLength={MAX_CHARS}
                  rows={7}
                />
                <span className="char-count">{message.length}<span> / {MAX_CHARS}</span></span>
              </div>

              <div className="context-heading"><span>MESSAGE CONTEXT</span><span>OPTIONAL</span></div>
              <div className="meta-row">
                <label className={`checkbox-label${forwarded ? ' is-selected' : ''}`}>
                  <input type="checkbox" checked={forwarded} onChange={e => setForwarded(e.target.checked)} />
                  <span className="checkbox-box" aria-hidden="true">
                    {forwarded && <svg width="12" height="12" viewBox="0 0 12 12"><path d="m2 6 2.5 2.5L10 3" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>}
                  </span>
                  <span className="checkbox-copy"><strong>Forwarded message</strong><small>Mark if it was forwarded</small></span>
                </label>

                <label className="sender-field">
                  <span>SENDER</span>
                  <select className="select" value={source} onChange={e => setSource(e.target.value)} aria-label="Sender">
                    <option value="unknown">Unknown sender</option>
                    <option value="friend">Friend / Contact</option>
                    <option value="group">Group message</option>
                  </select>
                </label>
              </div>
                </>
              ) : (
                <>
                  <label className="field-label" htmlFor="article-title">Article headline <span>REQUIRED</span></label>
                  <input id="article-title" className="article-input" value={articleTitle} onChange={e => setArticleTitle(e.target.value)} maxLength={500} placeholder="Enter the article headline..." />
                  <label className="field-label article-body-label" htmlFor="article-body">Article body <span>REQUIRED</span></label>
                  <textarea id="article-body" className="textarea article-body" value={articleBody} onChange={e => setArticleBody(e.target.value)} maxLength={20000} rows={9} placeholder="Paste the article text to review..." />
                  <div className="article-note">Historical dataset labels are not a fact-check. Model scores are not probabilities that an article is true or false.</div>
                </>
              )}

              <div className="form-actions">
                {mode === 'message' ? (
                  <button className="btn-primary" onClick={analyze} disabled={loading || !message.trim()}>
                    <span>{loading ? 'Analyzing message...' : 'Run security scan'}</span>
                    {!loading && <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 10h12m-5-5 5 5-5 5" /></svg>}
                  </button>
                ) : (
                  <button className="btn-primary" onClick={analyzeArticle} disabled={articleLoading || !articleTitle.trim() || !articleBody.trim()}>
                    <span>{articleLoading ? 'Reviewing article...' : 'Run article review'}</span>
                    {!articleLoading && <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 10h12m-5-5 5 5-5 5" /></svg>}
                  </button>
                )}
                {(result || error || articleResult || articleError) && <button className="btn-secondary" onClick={reset}>Clear</button>}
              </div>
              <div className="privacy-note">
                <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="4" y="8.5" width="12" height="9" rx="2" /><path d="M7 8.5V6a3 3 0 0 1 6 0v2.5" /></svg>
                <span>{mode === 'message'
                  ? 'Prediction runs on your local API. Text is sent to Groq only if you request AI advice.'
                  : 'Article text is analyzed by your local API. No live web evidence search is performed.'}</span>
              </div>
            </section>

            <section className={`panel intelligence-panel${result ? ' has-result' : ''}`} ref={resultRef}>
              {mode === 'article' ? (
                articleLoading ? (
                  <div className="loading-state"><div className="scan-visual"><span /><span /><span /></div><div className="section-kicker">ARTICLE REVIEW IN PROGRESS</div><h2>Comparing text patterns</h2><p>Applying separate models trained on historical article labels and headline annotations.</p></div>
                ) : articleError ? (
                  <div className="error-state" role="alert"><span className="state-icon error-icon">!</span><div className="section-kicker">REVIEW UNAVAILABLE</div><h2>Couldn’t review this article</h2><p>{articleError}</p><button className="btn-secondary" onClick={analyzeArticle}>Try again</button></div>
                ) : articleResult ? (
                  <div className="article-result">
                    <div className="section-kicker">HISTORICAL CORPUS PATTERN MATCH</div>
                    <h2>Article review</h2>
                    <div className="article-result-item"><span>VERACITY DATASET CATEGORY</span><strong>{articleResult.veracity_category}</strong><small>Model score: {Math.round(articleResult.veracity_model_score * 100)}% · historical label-pattern fit, not truth probability</small></div>
                    <div className="article-result-item"><span>HEADLINE CLICKBAIT LEVEL</span><strong>{articleResult.clickbait_category}</strong><small>Model score: {Math.round(articleResult.clickbait_model_score * 100)}% · historical annotation-pattern fit</small></div>
                    <div className="article-evaluation">
                      {[
                        ['article_veracity', 'Veracity'],
                        ['clickbait', 'Clickbait'],
                      ].map(([key, label]) => {
                        const metrics = articleResult.evaluation?.[key]
                        return metrics && (
                          <div key={key}><strong>{label} evaluation</strong><span>{metrics.split_method.replaceAll('_', ' ')} · macro-F1 {Math.round(metrics.test_macro_f1 * 100)}% (majority baseline {Math.round(metrics.majority_baseline_macro_f1 * 100)}%)</span></div>
                        )
                      })}
                    </div>
                    <p className="article-disclaimer">These models reflect limited, older datasets. They do not verify claims or establish whether this article is true. Use independent, current sources.</p>
                  </div>
                ) : (
                  <div className="empty-overview"><div className="overview-top"><div className="section-kicker">ARTICLE REVIEW</div></div><h2>Context, not a verdict.</h2><p>Review the headline and article body against historical training labels. Results are experimental signals, not live fact-checks.</p><div className="overview-divider" /><div className="model-footnote"><span className="model-indicator" /> ARTICLE VERACITY <span className="footnote-divider" /> HEADLINE CLICKBAIT</div></div>
                )
              ) : loading ? (
                <div className="loading-state">
                  <div className="scan-visual"><span /><span /><span /></div>
                  <div className="section-kicker">ANALYSIS IN PROGRESS</div>
                  <h2>Checking the signals</h2>
                  <p>Comparing message patterns against the trained spam model.</p>
                </div>
              ) : error ? (
                <div className="error-state" role="alert">
                  <span className="state-icon error-icon">!</span>
                  <div className="section-kicker">SCAN UNAVAILABLE</div>
                  <h2>Couldn’t complete the scan</h2>
                  <p>{error}</p>
                  <button className="btn-secondary" onClick={analyze}>Try again</button>
                </div>
              ) : result ? (
                <div className={`result-content result-${result.prediction.toLowerCase()}`} style={{ '--risk-color': result.riskColor, '--score-progress': `${result.score}%` }}>
                  <div className="result-overline"><span className="result-state-dot" /> SECURITY ASSESSMENT <span className="result-time">JUST NOW</span></div>
                  <div className="result-main">
                    <div className="result-copy">
                      <div className="section-kicker">SCAN COMPLETE</div>
                      <h2>{title}</h2>
                      <p>{summary}</p>
                    </div>
                    <div className="score-gauge" aria-label={`Spam score ${result.score} out of 100`}>
                      <div className="gauge-inner"><strong>{result.score}</strong><span>/ 100</span></div>
                    </div>
                  </div>
                  <div className="result-meta">
                    <div><span>RISK LEVEL</span><strong>{result.riskLevel}</strong></div>
                    <div><span>CLASSIFICATION</span><strong>{result.prediction}</strong></div>
                  </div>
                  <div className="signals-block">
                    <div className="signals-heading"><h3>Detected signals</h3><span>{result.reasons.length} FOUND</span></div>
                    <div className="signal-list">
                      {result.reasons.map((r, i) => (
                        <div className="signal-item" key={i}>
                          <span className="signal-mark" aria-hidden="true"><svg viewBox="0 0 16 16" fill="none"><path d="M8 1.7 13 3.5v3.7c0 3.2-2.2 5.8-5 7.1-2.8-1.3-5-3.9-5-7.1V3.5l5-1.8Z" /><path d="M8 5v3.2m0 2.2h.01" /></svg></span>
                          <span>{r}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="model-note"><span className="model-indicator" /> MODEL <strong>{result.modelUsed}</strong><span className="score-disclaimer">Score is heuristic, not a probability.</span></div>
                  <div className="result-report-action">
                    <button className="report-resources-button" type="button" onClick={() => setShowReport(true)}>
                      Find official reporting resources <span aria-hidden="true">↗</span>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="empty-overview">
                  <div className="overview-top">
                    <div className="section-kicker">YOUR SECURITY OVERVIEW</div>
                    <span className="overview-orbit" aria-hidden="true">
                      <svg viewBox="0 0 72 72" fill="none"><circle cx="36" cy="36" r="31" /><circle cx="36" cy="36" r="23" /><path d="M36 15 51 20v11c0 9.7-6.5 16.9-15 20.5C27.5 47.9 21 40.7 21 31V20l15-5Z" /><path d="m30 32 4 4 8-8" /></svg>
                    </span>
                  </div>
                  <h2>Be sure before<br />you trust.</h2>
                  <p>Run a scan to see the risk level, spam score, and the message signals behind the result.</p>
                  <div className="overview-divider" />
                  <div className="overview-checklist">
                    <div><span className="check-number">01</span><span><strong>Paste the message</strong><small>Text you want to inspect</small></span><span className="check-arrow">↗</span></div>
                    <div><span className="check-number">02</span><span><strong>Review the signals</strong><small>Understand what triggered the score</small></span><span className="check-arrow">↗</span></div>
                    <div><span className="check-number">03</span><span><strong>Choose what to do</strong><small>Use your judgment before acting</small></span><span className="check-arrow">↗</span></div>
                  </div>
                  <div className="model-footnote"><span className="model-indicator" /> TF-IDF CLASSIFIER <span className="footnote-divider" /> RULE-BASED SIGNALS</div>
                </div>
              )}
            </section>
          </div>

          {result && !loading && result.prediction !== 'BENIGN' && (
            <section className="groq-panel">
              <button
                className="groq-header"
                onClick={() => {
                  if (!advice && !adviceLoading && !adviceOpen) fetchAdvice()
                  else setAdviceOpen(x => !x)
                }}
              >
                <span className="groq-icon" aria-hidden="true">✳</span>
                <span className="groq-heading-copy"><strong>AI Security Advisor</strong><small>Optional analysis powered by Groq</small></span>
                <span className="groq-get">{adviceOpen ? 'CLOSE' : advice ? 'VIEW ADVICE' : 'GET ADVICE'} <b>↗</b></span>
              </button>

              {adviceOpen && (
                <div className="groq-body">
                  {adviceLoading && <div className="groq-loading-wrap"><div className="groq-spinner" /><span className="groq-loading-text">Consulting AI security advisor...</span></div>}
                  {adviceError && (
                    <div className="groq-error" role="alert">
                      <span>{adviceError}</span>
                      {!adviceLoading && <button className="btn-secondary" onClick={fetchAdvice}>Retry</button>}
                    </div>
                  )}
                  {advice && !adviceLoading && (
                    <>
                      {advice.verdict && <div className="groq-verdict">{advice.verdict}</div>}
                      {advice.reasoning && <><div className="groq-section-title">ANALYSIS</div><div className="groq-reasoning">{advice.reasoning}</div></>}
                      {advice.actions?.length > 0 && (
                        <><div className="groq-section-title">RECOMMENDED ACTIONS</div><div className="groq-actions">
                          {advice.actions.map((a, i) => <div className="groq-action-item" key={i}><div className="groq-action-num">{i + 1}</div><div className="groq-action-text">{a}</div></div>)}
                        </div></>
                      )}
                      {advice.report_to && (
                        <><div className="groq-section-title">WHERE TO REPORT</div><button className="groq-report" onClick={() => setShowReport(true)}>View reporting channels <span>↗</span></button></>
                      )}
                      <div className="groq-disclaimer">AI-generated advice may be inaccurate. Use your judgment.</div>
                    </>
                  )}
                </div>
              )}
            </section>
          )}

          {showReport && (
            <div className="popup-backdrop" onClick={() => setShowReport(false)}>
              <div className="popup" role="dialog" aria-modal="true" aria-labelledby="report-title" onClick={e => e.stopPropagation()}>
                <div className="popup-head">
                  <div>
                    <div className="popup-title" id="report-title">Report suspicious messages</div>
                    <div className="popup-sub">{selectedCountry ? `${selectedCountry.name} reporting resources` : 'Choose a country for local reporting resources'}</div>
                  </div>
                  <button className="popup-close" type="button" onClick={() => setShowReport(false)} aria-label="Close">×</button>
                </div>
                <div className="country-picker">
                  <label className="location-label" htmlFor="report-country">Country</label>
                  <div className="location-controls">
                    <select
                      id="report-country"
                      className="report-country-select"
                      value={countryCode}
                      onChange={event => {
                        setCountryCode(event.target.value)
                        setLocationStatus('')
                      }}
                    >
                      <option value="">Select a supported country</option>
                      {REPORTING_COUNTRIES.map(country => (
                        <option key={country.code} value={country.code}>{country.name}</option>
                      ))}
                    </select>
                    <button
                      className="btn-secondary detect-location-button"
                      type="button"
                      onClick={detectCountry}
                      disabled={locationLoading}
                    >
                      {locationLoading ? 'Detecting...' : 'Detect country'}
                    </button>
                  </div>
                  <p className="location-privacy">
                    Detection is optional and asks for browser permission only when clicked.
                    Coordinates are checked locally in this browser; they are not sent to a
                    location service, API, or saved.
                  </p>
                  {locationStatus && <p className="location-status" role="status">{locationStatus}</p>}
                </div>

                {selectedCountry ? (
                  <>
                    <div className="report-resource-heading">OFFICIAL RESOURCES FOR {selectedCountry.name.toUpperCase()}</div>
                    {countryResources.map(resource => (
                      <div className="helpline" key={resource.org}>
                        <div className="helpline-org">{resource.org}</div>
                        {resource.phone && (
                          <div className="helpline-row">
                            Phone: <a href={resource.phone.href}>{resource.phone.label}</a>
                          </div>
                        )}
                        {resource.action && <div className="helpline-row">{resource.action}</div>}
                        {resource.url && (
                          <div className="helpline-row">
                            <a href={resource.url} target="_blank" rel="noopener noreferrer">
                              Open official reporting page
                            </a>
                          </div>
                        )}
                        <div className="helpline-note">{resource.note}</div>
                      </div>
                    ))}
                  </>
                ) : (
                  <p className="location-empty">
                    Select one of the currently supported countries or use device location. If
                    your country is not listed, contact your local telecom provider or official
                    cybercrime authority.
                  </p>
                )}

                <div className="report-resource-heading">WHATSAPP REPORTING</div>
                {GLOBAL_REPORTING_RESOURCES.map(resource => (
                  <div className="helpline" key={resource.org}>
                    <div className="helpline-org">{resource.org}</div>
                    <div className="helpline-note">{resource.note}</div>
                  </div>
                ))}
                <div className="popup-foot">
                  Sources checked 3 November 2026. Resources vary by country and may change; verify
                  details on the official site. This directory does not submit a report for you.
                </div>
              </div>
            </div>
          )}
        </main>

        <a className="learn-link" href="/how-it-works">
          <span><strong>How does the model work?</strong><small>Explore the data, model pipeline, scoring rules, and limitations.</small></span>
          <span className="learn-link-arrow" aria-hidden="true">→</span>
        </a>
        <footer className="footer">
          <span>MESSAGE RISK ANALYZER</span>
          <span>Separate message and article models <i /> Experimental signals — verify important claims independently</span>
        </footer>
      </div>
    </div>
  )
}
