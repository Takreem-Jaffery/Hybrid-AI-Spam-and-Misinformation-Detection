import { useState, useEffect } from 'react'
import { applyTheme } from './theme.js'
import './styles.css'

export default function HowItWorks() {
  const [theme, setTheme] = useState('light')

  useEffect(() => { applyTheme(theme) }, [theme])

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
            <a className="back-link" href="/">Back to scanner <span aria-hidden="true">→</span></a>
            <button className="theme-toggle" onClick={() => setTheme(t => (t === 'light' ? 'dark' : 'light'))} title="Toggle theme" aria-label="Toggle theme">
              <span aria-hidden="true">{theme === 'light' ? '◐' : '◑'}</span>
            </button>
          </div>
        </header>

        <main className="method-page">
          <div className="eyebrow"><span className="eyebrow-line" /> MODEL TRANSPARENCY</div>
          <h1>How the analysis<br /><span>works</span></h1>
          <p className="method-intro">A practical overview of the independent message and article analysis tasks, their training data, and their limits.</p>

          <section className="method-note">
            <strong>Scope matters</strong>
            <p>The message scanner and article review are separate tasks with separate data and models. Article review compares text patterns to historical veracity and headline annotations; it does not verify claims or search current reporting.</p>
          </section>

          <section className="method-section">
            <div className="section-kicker">THE CURRENT PIPELINE</div>
            <h2>From text to a risk assessment</h2>
            <div className="pipeline">
              <article className="pipeline-step">
                <span className="pipeline-number">01</span>
                <div><h3>Message and context</h3><p>The web interface accepts up to 500 characters. Users may also mark whether the message was forwarded and whether its sender is unknown, a friend, or a group. The API accepts up to 2,000 characters.</p></div>
              </article>
              <article className="pipeline-step">
                <span className="pipeline-number">02</span>
                <div><h3>Text preparation</h3><p>For TF-IDF, text is lowercased, URLs and numbers are replaced with tokens, punctuation is removed, and whitespace is normalized. The same preprocessing function is shared by training and inference.</p></div>
              </article>
              <article className="pipeline-step">
                <span className="pipeline-number">03</span>
                <div><h3>Spam classification</h3><p>The primary model is a TF-IDF text pipeline with up to 5,000 unigram and bigram features. Naive Bayes, logistic regression, and random forest are compared; the deployed model is selected by five-fold cross-validation on the training split.</p><p>If a locally trained DistilBERT model is installed, it can also score the raw message and be blended with the TF-IDF score. It is optional and is not included in a fresh core install.</p></div>
              </article>
              <article className="pipeline-step">
                <span className="pipeline-number">04</span>
                <div><h3>Rules and score</h3><p>When both classifiers are available, their scores are combined as 65% TF-IDF and 35% DistilBERT. With the standard install, only the TF-IDF model is used. Hand-set context bonuses then add 8 points for a forwarded message and 7 for an unknown sender. A narrow OTP/urgency rule may impose a minimum score of 55.</p><p>Spam is assigned above 50. Risk is Low below 40, Medium from 40 to below 65, and High from 65. The score is clipped to 0–100 and is not a calibrated probability. Other listed signals explain the result; they do not each add points to the score.</p></div>
              </article>
              <article className="pipeline-step">
                <span className="pipeline-number">05</span>
                <div><h3>Explanation and optional advice</h3><p>The app lists rule-based signals such as suspicious words, URLs, long numbers, excess capitals, and message context. If configured with a Groq API key, the user can separately request LLM advice; it is not part of the classifier or its accuracy metrics.</p></div>
              </article>
            </div>
          </section>

          <section className="method-section method-data">
            <div className="section-kicker">TRAINING DATA</div>
            <h2>Separate datasets for separate tasks</h2>
            <p>The message spam model uses the English SMS Spam Collection: 5,572 messages (4,825 ham and 747 spam) from a UK/Singapore collection. Its evaluation uses a stratified 80/20 split with seed 42.</p>
            <div className="data-facts">
              <div><strong>5,572</strong><span>English SMS messages</span></div>
              <div><strong>80 / 20</strong><span>Stratified train / test split</span></div>
              <div><strong>42</strong><span>Reproducibility seed</span></div>
            </div>
            <p className="method-caveat">The collection is small and dates from the 2000s, so it may not represent modern scams or every kind of message. Treat the result as an aid, not a guarantee.</p>
          </section>

          <section className="method-section">
            <div className="section-kicker">ARTICLE REVIEW</div>
            <h2>Historical labels, not a fact-check</h2>
            <p>The veracity classifier preserves the four categories in the BuzzFeed-Webis 2016 dataset: mostly true, mixture of true and false, mostly false, and no factual content. The headline classifier predicts one of four Webis Clickbait-17 severity levels. It maps each mean human rating to the nearest original severity level. These are separate models and evaluation reports.</p>
            <p>The BuzzFeed-Webis corpus contains 1,627 fact-checked articles; Webis Clickbait-17 contains 38,517 annotated posts overall. Clickbait training uses the provided train split and evaluates on its separate validation split. The veracity task holds out publisher groups; full per-class metrics and dataset checksums are generated with the training scripts.</p>
            <p>In the first CPU baseline, veracity macro-F1 was 26.32% versus a 19.61% majority baseline; the held-out model had zero F1 for two rare verdict classes. Clickbait macro-F1 was 35.58% versus a 14.92% majority baseline. These limited results are shown in Article Review and do not support using the veracity output as a reliable fact-check.</p>
            <p className="method-caveat">Both sources are limited and dated. Their model scores describe fit to historical annotation patterns, not the probability that a current article is true. Article Review does not perform live evidence lookup; verify important claims with current, independent sources.</p>
          </section>

          <div className="method-cta"><a className="btn-primary" href="/">Return to analyzer <span aria-hidden="true">→</span></a></div>
        </main>

        <footer className="footer">
          <span>MESSAGE RISK ANALYZER</span>
          <span>Experimental model — verify important messages independently</span>
        </footer>
      </div>
    </div>
  )
}
