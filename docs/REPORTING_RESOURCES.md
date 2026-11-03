# Reporting resources and location privacy

The directory is an initial, curated set of official resources—not a complete global support
service. Sources were checked on 3 November 2026. A listed site may be unavailable, change its
eligibility rules, or not accept every type of report. The app does not send reports on the user's
behalf.

## Country coverage

| Country | Official source | Scope and caveats |
|---|---|---|
| Australia | [Scamwatch](https://www.scamwatch.gov.au/report-a-scam), [ACMA scam-call/text guidance](https://www.acma.gov.au/report-scam-phone-call-or-text-message) | Scam and telecom reporting guidance. |
| Canada | [Canadian Anti-Fraud Centre](https://antifraudcentre-centreantifraude.ca/report-signalez-eng.htm), [Spam Reporting Centre](https://fightspam-combattrelepourriel.ised-isde.canada.ca/site/canada-anti-spam-legislation/en/form/report-spam-form) | Fraud/cybercrime and spam/electronic-threat reporting are separate channels. The CAFC says victims should also contact local police. |
| Germany | [Federal Network Agency consumer portal](https://www.bundesnetzagentur.de/DE/Vportal/TK/Aerger/start.html) | Telecom complaints and warnings; a central nationwide online-fraud intake was not verified. |
| India | [Sanchar Saathi—Chakshu](https://sancharsaathi.gov.in/sfc/), [National Cyber Crime Reporting Portal](https://www.cybercrime.gov.in/) | Chakshu covers suspected fraudulent calls, SMS, and WhatsApp communications and unsolicited commercial messages. The 1930 number is for financial cyber fraud after money has been lost. |
| Indonesia | [Aduan Nomor (Komdigi)](https://aduannomor.id/) | Complaints about phone numbers associated with fraud or misuse. |
| Japan | [National Police Agency cyber-incident consultation](https://www.npa.go.jp/bureau/cyber/soudan.html) | National guidance and routes to prefectural police; a dedicated nationwide SMS-spam channel was not verified. |
| Pakistan | [PTA complaints](https://complaint.pta.gov.pk/RegisterComplaint.aspx), [FIA cybercrime complaints](https://complaint.fia.gov.pk/) | Official telecom and cybercrime portals. Both sites were unavailable during the source check, so confirm availability before relying on them. |
| South Korea | [National Police cybercrime system](https://ecrm.police.go.kr/minwon/main), [KISA](https://www.kisa.or.kr/) | Police cybercrime reporting and KISA smishing assistance; 118 is listed for KISA assistance. |
| Türkiye | [BTK consumer complaint system](https://tuketicisikayet.btk.gov.tr/), [Cyber Security Directorate](https://siberguvenlik.gov.tr/) | BTK is a telecom-consumer channel. A current nationwide public channel specifically for fraud SMS or online-fraud victims was not verified. |
| United Kingdom | [GOV.UK phishing guidance](https://www.gov.uk/report-suspicious-emails-websites-phishing), [Report Fraud](https://www.reportfraud.police.uk/reporting-a-fraud) | Forward suspicious texts to 7726 and emails to `report@phishing.gov.uk`. Report Fraud and its 0300 123 2040 number apply to England and Wales; the guidance directs people in Scotland to Police Scotland. |
| United States | [FTC ReportFraud](https://reportfraud.ftc.gov/) | Report fraud, scams, phishing, and text scams. |

WhatsApp's in-app report/block instructions are shown independently of country. Where the official
source describes a narrower purpose than general fraud reporting, the UI says so rather than
presenting it as a universal hotline. Phone numbers are included only when verified on an official
source and are displayed with their specific scope.

## Location handling

- The user chooses **Detect country** after opening the reporting panel. This invokes the browser's
  Geolocation API and requires explicit browser permission.
- Latitude/longitude are checked in memory against bundled Natural Earth country boundaries
  distributed by [`world-atlas`](https://github.com/topojson/world-atlas) (ISC package; the
  underlying Natural Earth data is public domain). Country matching runs locally; there is no
  reverse-geocoding or IP-location request.
- The app keeps only the selected country code in React state for the current page session. It
  does not send or persist coordinates. Manual selection is available if permission is declined,
  detection fails, or the user's country is not in this first set.
- The 1:110m boundary dataset is a small-scale approximation. Border areas and territories may be
  misclassified; users can override the result. Country selection is not legal-jurisdiction
  advice.
- Geolocation is available only in secure contexts such as HTTPS or localhost. A deployment must
  use HTTPS for device detection; the manual country selector works without it.

When adding countries, use primary government, regulator, or national cyber-security sources;
verify each phone/SMS instruction on that source; record the check date and scope; and show a
manual fallback. Do not silently substitute IP lookup or send precise coordinates to a third
party.
