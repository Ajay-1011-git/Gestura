# Domain glossary — `medical`

Selected when Setu is deployed at a clinic or hospital desk. This is the domain
the PRD's host-site user story names directly: staff want domain vocabulary to
resolve correctly more often than a general-purpose pass manages.

**Scope is a front desk, not a consultation.** Every term here belongs to
reception, triage and wayfinding — the conversation that happens *before* a
clinician is in the room. Diagnostic language, drug names, dosages and anything
that carries clinical consequence are deliberately absent, and the medical/legal
boundary gate that would be needed to handle them safely is Stage 3 work that
does not exist yet. Setu is not a certified medical interpreter and this file is
written so it cannot be mistaken for one.

**A wrong hit here is the expensive failure.** "Patient" and "patience" are one
keystroke apart in a transcript and unrelated in ISL; lookup is exact-match for
that reason, and a term whose correct sign depends on clinical context that Setu
cannot see is left out entirely rather than guessed.

Reviewed by hand, same as T1.2's vocabulary manifest. A gloss here is not a
promise the sign renders — see `general.md` for what a gloss does and does not
guarantee.

| Term | Gloss | Note |
|---|---|---|
| hospital | HOSPITAL | In the Stage 1 lexicon. |
| clinic | HOSPITAL | Same sign in ISL; the distinction is carried by context, not a separate sign. |
| doctor | DOCTOR | |
| nurse | NURSE | |
| patient | PATIENT | The person, never the adjective. Exact-match only, for that reason. |
| appointment | APPOINTMENT | |
| emergency | EMERGENCY | |
| ambulance | AMBULANCE | |
| pain | PAIN | Presence of pain only. Severity and location are not glossed here — they need a clinician, not a glossary. |
| medicine | MEDICINE | The general category. Specific drug names are out of scope and fall through to fingerspelling. |
| fever | FEVER | |
| blood | BLOOD | |
| test | TEST | Covers "blood test", "lab test". Not a result. |
| report | REPORT | The document, e.g. collecting a test report at the desk. |
| reception | RECEPTION | |
| waiting room | WAIT ROOM | Two signs; rendered in sequence. |
| ward | WARD | |
| insurance | INSURANCE | |
| form | FORM | The paperwork sense. |
| help | HELP | In the Stage 1 lexicon. Carried from `general` on purpose — it is the single highest-frequency term at a desk. |
| where | WHERE | Carried from `general`: wayfinding is most of what this desk is asked. |
| today | TODAY | Carried from `general`: appointment scheduling. |
