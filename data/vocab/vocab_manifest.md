# Gestura — Curated ISL Vocabulary Manifest (Stage 1)

> **STATUS: DRAFT — PENDING PROJECT OWNER REVIEW.**
> T1.2 gates T1.3, T1.4, and T1.9. Do not begin pose extraction against this
> list until the owner has cut, edited, and approved it. Architecture v3 §11.1
> calls fixing this list "the single highest-leverage protective decision for
> the whole project."

## What this file is

The closed, finite set of ISL signs Gestura will recognize (Sign→Speech) and
produce (Speech→Sign) for the Stage 1 demo. FR-1 forbids the system claiming
recognition confidence for anything outside this list. Every other recognition
and lookup task in Stage 1 reads from here.

## Honesty note — read before trusting the `notes` column

Per NFR-5 and Risk R-3, no Deaf-community validation is available within this
timeline. The **`gloss_id`** and **English meaning** columns are labels and are
safe to treat as a design decision. The **`notes`** column is a hearing
non-signer's description of articulation drawn from general public reference,
and is **not validated ISL**. Treat every note as a hypothesis to be confirmed
against a real ISL reference or a Deaf signer before it is rehearsed, and
re-record any sign whose note turns out to be wrong. Do not present these notes
to judges as evidence of linguistic correctness — that is precisely the
overclaim R-7 warns against.

## Scope shape

Chosen for one narrow context — **a hospital reception desk** — matching the
PRD's driving scenario. 30 entries: small enough to rehearse and re-record in
an afternoon, broad enough to form real sentences in both directions.

---

## Vocabulary

| gloss_id | English meaning | notes |
|---|---|---|
| HELLO | hello / greetings | Opens demo scene 1. Single clear movement, should be one of the most reliably recognized entries. |
| THANK-YOU | thank you | Natural demo closer. Verify articulation against an ISL reference. |
| PLEASE | please | Politeness marker; may be optional in ISL grammar — confirm whether it is normally signed or omitted before relying on it. |
| SORRY | sorry / excuse me | Useful opener for an unscheduled walk-up interaction. |
| YES | yes | Short sign. Short signs give the DTW classifier less signal — expect lower confidence and check this empirically in T1.4. |
| NO | no | Same short-sign caveat as YES. |
| ME | I / me | Pointing sign (indexical). Trajectory is short and may be confusable with other pointing signs — candidate for the ambiguity pair, see below. |
| YOU | you | Pointing sign directed outward. Differs from ME mainly in direction, not handshape — strong candidate for the ambiguity pair. |
| DOCTOR | doctor | Core noun for this context. |
| NURSE | nurse | Include only if visually distinct enough from DOCTOR to be worth the extra recording; owner's call. |
| HOSPITAL | hospital | Core noun for this context. |
| APPOINTMENT | appointment / booking | Central to the reception-desk scenario. |
| MEDICINE | medicine / tablet | Core noun. |
| PAIN | pain / hurt | Often accompanied by non-manual facial marking in ISL that the Stage 1 pipeline does not capture — a real, disclosable limitation. |
| HELP | help | High-value verb for this context. |
| WANT | want | Used in demo scene 1's example sentence (`ME DOCTOR MEET WANT`). |
| NEED | need | Semantically close to WANT; confirm the two are visually distinct before recording both. |
| MEET | meet | Used in demo scene 1's example sentence. |
| WAIT | wait | Directly useful for the collision/hold scene's narrative. |
| GO | go | Basic directional verb. |
| COME | come | Reverse of GO; likely differs mainly in direction — second candidate for the ambiguity pair. |
| HAVE | have | Basic possessive verb. |
| WHAT | what | Question word. ISL commonly places question words at the end of the clause — this affects T1.8's gloss ordering, not just recognition. |
| WHERE | where | Question word; same clause-position note as WHAT. |
| WHO | who | Question word; same clause-position note as WHAT. |
| WHEN | when | Question word; same clause-position note as WHAT. |
| TODAY | today | Time marker. ISL typically establishes time at the start of a clause — relevant to T1.8's gloss ordering. |
| TOMORROW | tomorrow | Time marker; may share a trajectory family with TODAY, so check their DTW distance in T1.4. |
| NAME | name | Needed to set up the fingerspelling/refusal scene, since a person's name follows it. |
| UNDERSTAND | understand | Lets the signer confirm or deny comprehension — directly supports the clarification scene's resume path. |

---

## Entries deliberately excluded, and why

These are **not** oversights. Each is excluded to make a specific demo scene
work honestly.

| Excluded term | Why it stays out |
|---|---|
| Any personal name (e.g. a patient's name) | Must fall through to fingerspelling to satisfy FR-7 and demo scene 4. Adding a name sign would destroy the refusal scene. |
| Any specific drug name (e.g. a branded medication) | Same as above — the intended out-of-vocabulary trigger for the `FINGERSPELLING` / `UNMATCHED` coverage path in T1.9. |
| Numbers | ISL number handshapes are a system of their own, not 10 isolated signs. Recording them properly is a larger job than it looks, and nothing in the four Stage 1 demo scenes requires them. Defer unless the owner wants a number in the script. |
| Fingerspelled alphabet handshapes | `spoken-to-signed-translation` supplies fingerspelling fallback as built-in behavior (architecture v3 §6.8) — it should not be reimplemented as vocabulary entries. |

## Planned out-of-vocabulary trigger for demo scene 4

Pick **one** proper noun and rehearse it, so the refusal scene is deterministic
rather than improvised. Owner to choose — a patient name is the most natural
fit for a reception desk, and pairs with the in-vocabulary `NAME` sign to set it
up (`NAME` + *[fingerspelled name]*).

---

## Open decisions for the owner

1. **Cut to fit rehearsal time.** 30 entries is an upper bound, not a target.
   Every entry needs at least 3–5 recorded takes for T1.3/T1.4. If that is more
   recording than the timeline allows, cut from the bottom: `NURSE`, `NEED`,
   `WHO`, `WHEN`, `TOMORROW`, `HAVE` are the least load-bearing for the four
   Stage 1 demo scenes.
2. **Do not pre-pick the ambiguity pair.** Demo scene 2 needs two signs the
   classifier *genuinely* confuses, so the low confidence is real rather than
   staged — FR-2 requires a calibrated confidence value, not a placeholder.
   Confusability here is a property of **pose trajectory similarity**, which is
   what T1.4's DTW distance actually measures — not of linguistic similarity,
   and not something to guess in advance. After T1.3 extraction, compute the
   pairwise DTW distance matrix across all entries and pick the closest real
   pair. `ME`/`YOU` and `GO`/`COME` are flagged above as likely candidates, but
   the measurement decides, not this document.
3. **Confirm the articulation notes.** See the honesty note at the top. The
   `notes` column is unvalidated and should be corrected against a real ISL
   reference before recording begins.
4. **Decide on `PLEASE`.** If it is not naturally signed in ISL for this
   context, drop it rather than record an artificial sign.

## Recording checklist (feeds T1.3)

- One subdirectory per `gloss_id` under `data/vocab/raw_video/`.
- 3–5 takes per sign, varying speed slightly, so T1.4 has a real distance
  distribution to calibrate confidence against rather than a single exemplar.
- Fixed camera position, consistent lighting, upper body and both hands fully
  in frame for the whole sign.
- Record one clean take of the chosen out-of-vocabulary name too — not as
  vocabulary, but so the refusal scene can be rehearsed end to end.
