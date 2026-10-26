# Medicine sheet: ind-mr-sangraha-v (avnishbm)

Interim first-150 submission. Labels reflect the user’s full-document decisions. The remaining 350 documents, independent second review, and approved-gold harm tests are pending.

Approved Marathi gold from Data A/B is expected on 29 October 2026 and has not been received. No gold harm-test pass or reviewer agreement is claimed. Final medicine results are due 31 October 2026, 23:59 IST.

Counts below are documents, not occurrences, in this 150-row submission.

## C1: 30 documents

**Examples:**

- `e14f9a0e6e8d1d9b`: C1: "Read Goa news in Marathi" at offset 741
- `e9320e666a90a129`: C1: "Read the latest agriculture news in Marathi" at offset 1069
- `fab4928039a870ab`: C1: "Recover your password.\n A password will be e-mailed to you." at offset 211

**Proposed medicine:** remove an exact, reviewed terminal publisher/UI suffix at a line boundary, leaving substantive article content. This narrow rule addresses only the listed footer layouts; other C1 cases require separate contextual rules.

**Action:** fix a copy; do not modify the source pack or drop the document.

```python
APPROVED_SUFFIXES = (' Read Goa news in Marathi and Goa local news on Tourism, Business, Politics, Entertainment, Sports and Goa latest news in Marathi on Dainik Gomantak. Get Goa news live updates on the Dainik Gomantak Mobile app for Android and IOS.\n दैनिक गोमन्तक आता सर्व सोशल मीडिया प्लॅटफॉर्मवर. ताज्या घडामोडींसाठी फेसबुक, इन्स्टाग्राम, ट्विटर, शेअर चॅट आणि टेलिग्राम आम्हाला फॉलो करा. तसेच, दैनिक गोमन्तकच्या यूट्यूब चॅनेलला आजच सबस्क्राइब करा. ', ' Read the latest agriculture news in Marathi and watch Agriculture videos on Agrowon. Get the latest updates on Market Intelligence, Market updates, Bazar Bhav, Agriculture Jugad, and Farmer Success Stories.\n ताज्या घडामोडींसाठी फेसबुक, ट्विटर, इन्स्टाग्राम आणि टेलिग्रामवर आम्हाला फॉलो करा. तसेच, एग्रोवनच्या यूट्यूब चॅनेलला आजच सबस्क्राइब करा. ', ' Recover your password.\n A password will be e-mailed to you. ')

def remove_footer(text):
    for suffix in APPROVED_SUFFIXES:
        if text.endswith(suffix):
            start = len(text) - len(suffix)
            if start > 0 and text[start - 1] == "\n" and text[:start].strip():
                return text[:start]
    return text
```

**Side effects:** a legitimate quotation of an entire listed footer at the end of an article could be shortened. Whitespace and wording variants evade exact matching. The rule does not address every C1 layout.

**Approved-gold harm test:** NOT RUN. Gold size, changed count, dropped count and changed-or-dropped union are pending. More than 1% affected fails. Synthetic regressions are not the required gold-set result.

## D3: 32 documents

**Examples:**

- `e14f9a0e6e8d1d9b`: D3: "Read Goa news in Marathi" at offset 741
- `a3ae0a2e9168261e`: D3: "किमुक्तम् । तत्र व्यपदेशिवद्वचनमेकाचो" at offset 0
- `e9320e666a90a129`: D3: "(Agriculture Produce Market Committee Election)" at offset 63

**Medicine status:** no blanket text-removal or translation rule is proposed for natural code-switching, quotations, terminology or glosses. D3 is descriptive under DISEASES.md. Reviewer-selected `fix` verdicts are preserved; the intended treatment of natural D3-only rows needs a consistent lab policy.

**Action pending:** retain legitimate mixed-language content while that policy is resolved. Publisher/UI text independently labelled C1 may be addressed by the C1 rule. No untested language filter is proposed.

**Side effects to avoid:** deleting English words or translating every English segment would damage legitimate names, quotations, technical terms and code-switching.

**Approved-gold harm test:** NOT RUN. Gold size, changed count, dropped count and changed-or-dropped union are pending. More than 1% affected fails. Synthetic regressions are not the required gold-set result.

## F1: 8 documents

**Examples:**

- `fab4928039a870ab`: F1: private-person address at offset 9
- `8924481cb3403322`: F1: private-person residential address at offset 333
- `3532a8bbeeb1e5fe`: F1: identifiable private-person medical information at offset 49

**Medicine status:** a general private-data rule is pending design and review. Redact only confirmed private spans in a copy; do not infer private status from every number, name, location or public event contact.

**Action pending:** span-level redaction or document rejection must be chosen per confirmed case. Current evidence records type and position only; it does not expose sensitive values or provide complete redaction ranges.

**Side effects to avoid:** a blanket address/number/name rule can destroy public information, historical references, dates, quantities and legitimate organization contacts.

**Approved-gold harm test:** NOT RUN. Gold size, changed count, dropped count and changed-or-dropped union are pending. More than 1% affected fails. Synthetic regressions are not the required gold-set result.
