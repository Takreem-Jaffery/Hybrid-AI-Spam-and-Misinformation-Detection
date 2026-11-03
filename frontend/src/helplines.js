// Pakistan reporting channels shown in the "Where to report" popup.
//
// Numbers were cross-checked against public sources on 2026-10-30 (PTA: phoneworld.com.pk
// "PTA consumer service short codes"; FIA: pakobserver.net CCW helpline article and
// propakistani.pk reporting guide). Official pages change, so confirm on pta.gov.pk and
// nr3c.gov.pk before relying on this. The phone numbers and e-mail addresses used in the first
// version of this project could not be verified and were removed.
export const HELPLINES = [
  {
    org: 'FIA Cyber Crime Wing (NR3C)',
    phone: '1991',
    url: 'https://complaint.fia.gov.pk/',
    urlLabel: 'complaint.fia.gov.pk',
    note: 'Federal cyber crime authority: online complaint form or helpline',
    color: '#3b82f6',
  },
  {
    org: 'PTA',
    phone: '0800-55055',
    sms: 'Forward spam SMS to 9000',
    url: 'https://www.pta.gov.pk/',
    urlLabel: 'pta.gov.pk',
    note: 'Pakistan Telecommunication Authority: SMS / phone spam',
    color: '#8b5cf6',
  },
  {
    org: 'WhatsApp',
    note: 'Inside WhatsApp: open the chat, tap the sender name, then Report (or block the contact)',
    color: '#22c55e',
  },
]
