export const REPORTING_COUNTRIES = [
  { code: 'AU', name: 'Australia' },
  { code: 'CA', name: 'Canada' },
  { code: 'DE', name: 'Germany' },
  { code: 'IN', name: 'India' },
  { code: 'ID', name: 'Indonesia' },
  { code: 'JP', name: 'Japan' },
  { code: 'PK', name: 'Pakistan' },
  { code: 'KR', name: 'South Korea' },
  { code: 'TR', name: 'Türkiye' },
  { code: 'GB', name: 'United Kingdom' },
  { code: 'US', name: 'United States' },
]

export const COUNTRY_REPORTING_RESOURCES = {
  AU: [
    {
      org: 'Scamwatch — Australian Competition and Consumer Commission',
      url: 'https://www.scamwatch.gov.au/report-a-scam',
      note: 'Report scams and online fraud to the ACCC.',
    },
    {
      org: 'Australian Communications and Media Authority',
      url: 'https://www.acma.gov.au/report-scam-phone-call-or-text-message',
      note: 'Guidance for reporting scam calls and text messages.',
    },
  ],
  CA: [
    {
      org: 'Canadian Anti-Fraud Centre',
      url: 'https://antifraudcentre-centreantifraude.ca/report-signalez-eng.htm',
      phone: { label: '1-888-495-8501', href: 'tel:18884958501' },
      note: 'Report fraud and cybercrime. If you are a victim, the Centre recommends contacting local police.',
    },
    {
      org: 'Spam Reporting Centre — Innovation, Science and Economic Development Canada',
      url: 'https://fightspam-combattrelepourriel.ised-isde.canada.ca/site/canada-anti-spam-legislation/en/form/report-spam-form',
      note: 'Report spam and other electronic threats. The online form explains how submitted information is handled.',
    },
  ],
  DE: [
    {
      org: 'Federal Network Agency (Bundesnetzagentur)',
      url: 'https://www.bundesnetzagentur.de/DE/Vportal/TK/Aerger/start.html',
      note: 'Telecommunications consumer complaints and warnings about scam calls or messages. For suspected criminal fraud, contact local police; a central nationwide fraud intake was not verified.',
    },
  ],
  IN: [
    {
      org: 'Sanchar Saathi — Chakshu',
      url: 'https://sancharsaathi.gov.in/sfc/',
      note: 'Report suspected fraudulent communications received by call, SMS, or WhatsApp, and unsolicited commercial messages. The portal says reports within 30 days can be investigated.',
    },
    {
      org: 'National Cyber Crime Reporting Portal',
      url: 'https://www.cybercrime.gov.in/',
      phone: { label: '1930 if money has been lost', href: 'tel:1930' },
      note: 'Report cybercrime and financial fraud. Use 1930 for financial cyber fraud after money is lost.',
    },
  ],
  ID: [
    {
      org: 'Aduan Nomor — Ministry of Communication and Digital Affairs (Komdigi)',
      url: 'https://aduannomor.id/',
      note: 'Report phone numbers associated with fraud or misuse, including suspicious calls or messages.',
    },
  ],
  JP: [
    {
      org: 'National Police Agency — cyber incident consultation',
      url: 'https://www.npa.go.jp/bureau/cyber/soudan.html',
      note: 'National guidance and online reporting routes for cyber incidents; reports are directed to prefectural police. A nationwide dedicated SMS-spam channel was not verified.',
    },
  ],
  PK: [
    {
      org: 'Pakistan Telecommunication Authority (PTA)',
      url: 'https://complaint.pta.gov.pk/RegisterComplaint.aspx',
      note: 'Official telecom complaint portal. The portal could not be reached during the latest source check; confirm its current complaint categories and availability before relying on it.',
    },
    {
      org: 'Federal Investigation Agency (FIA)',
      url: 'https://complaint.fia.gov.pk/',
      note: 'Official cybercrime complaint portal. The site returned a service error during the latest source check; verify availability before relying on it.',
    },
  ],
  KR: [
    {
      org: 'Korean National Police — Cyber Crime Reporting System',
      url: 'https://ecrm.police.go.kr/minwon/main',
      note: 'Police guidance and reporting for cybercrime, including fraud.',
    },
    {
      org: 'Korea Internet & Security Agency (KISA)',
      url: 'https://www.kisa.or.kr/',
      phone: { label: '118 for smishing assistance', href: 'tel:118' },
      note: 'KISA assistance for smishing and internet security incidents.',
    },
  ],
  TR: [
    {
      org: 'Information and Communication Technologies Authority (BTK)',
      url: 'https://tuketicisikayet.btk.gov.tr/',
      note: 'Telecommunications consumer complaint system. A current nationwide public channel specifically for fraud SMS or online-fraud victims was not verified.',
    },
    {
      org: 'Cyber Security Directorate',
      url: 'https://siberguvenlik.gov.tr/',
      note: 'Official cybersecurity information; use the telecom complaint system or local authorities for case reporting.',
    },
  ],
  GB: [
    {
      org: 'GOV.UK — report internet scams and phishing',
      url: 'https://www.gov.uk/report-suspicious-emails-websites-phishing',
      action: 'Forward suspicious texts to 7726 (free); forward suspicious emails to report@phishing.gov.uk.',
      note: 'For fraud in England and Wales, use Report Fraud. In Scotland, report to Police Scotland.',
    },
    {
      org: 'Report Fraud — England and Wales',
      url: 'https://www.reportfraud.police.uk/reporting-a-fraud',
      phone: { label: '0300 123 2040', href: 'tel:03001232040' },
      note: 'Use for online fraud if you are in England or Wales.',
    },
  ],
  US: [
    {
      org: 'Federal Trade Commission (FTC) — ReportFraud',
      url: 'https://reportfraud.ftc.gov/',
      note: 'Report fraud, scams, and suspicious business practices, including phishing and text scams.',
    },
  ],
}

export const GLOBAL_REPORTING_RESOURCES = [
  {
    org: 'WhatsApp — report in the app',
    note: 'In WhatsApp, open the chat, tap the sender or group name, then choose Report. Block the sender if appropriate.',
  },
]
