# Gestura — Curated ISL Vocabulary Manifest (Stage 1)

> **STATUS: SCOPED TO AVAILABLE TRAINING DATA — approved 2026-09-15.**
> T1.2 gates T1.3, T1.4, and T1.9. Architecture v3 §11.1 calls fixing this list
> "the single highest-leverage protective decision for the whole project."
>
> **REVISION 2026-09-15:** the original 30-entry hospital list was scoped down
> to the 16 entries below, chosen so that **every class has real training data
> already available** and no signs need to be recorded before the pipeline can
> run end to end. The 14 deferred entries are listed in "Phase 2 vocabulary"
> near the bottom — deferring them costs nothing structurally, because adding a
> class is a cheap retrain of the same small classifier head (see T1.4).

## Data source for this vocabulary

All 16 classes are covered by
[`vidit031/isl-isolated-40words`](https://huggingface.co/datasets/vidit031/isl-isolated-40words)
— 642 MP4 clips over 40 ISL glosses, **ungated**, last updated 2026-07-25,
aggregated from ISL500 (405 clips), INCLUDE (143), CISLR (81) and the ISLRTC
dictionary (13). It is a **derived aggregate under mixed upstream licenses**
(the card is explicit that it is *not* dual-licensed as one open license) —
respect each upstream license if any subset is redistributed, and cite the
originals. For a supervised pilot this is a usable research corpus, not a
cleared-for-production dataset; that distinction belongs in any public
description of the system, consistent with G-4.

**Selection rule: minimum 16 clips per class.** Four entries from the original
manifest were dropped purely because the corpus does not have enough data for
them — `when` (2 clips), `me` (3), `come` (4), `sorry` (6). Training and
holding out a test set on 2–6 examples would produce a confidence number that
is not calibrated, which FR-2 forbids. They move to Phase 2.

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

One narrow context — **a hospital reception desk**, matching the PRD's driving
scenario — narrowed further to what the corpus can actually support. 16 active
entries, 326 clips, minimum 16 clips per class.

---

## Active vocabulary (Stage 1)

`clips` is the real count in `vidit031/isl-isolated-40words`, read from its
`metadata.csv` on 2026-09-15.

| gloss_id | English meaning | notes |
|---|---|---|
| HELLO | hello / greetings | 36 clips — best-sampled class in the set. Opens demo scene 1; expect this to be among the most reliably recognized. |
| THANK-YOU | thank you | 39 clips — best-sampled class. Dataset label is `thank you`; the space must be normalized on load. Natural demo closer. |
| PLEASE | please | 18 clips. May be optional in ISL grammar — confirm whether it is normally signed or omitted rather than assuming English politeness maps across. |
| YES | yes | 17 clips. Short sign, so fewer frames of signal; watch its confidence distribution in T1.4 rather than assuming it behaves like longer signs. |
| NO | no | 17 clips. Same short-sign caveat as YES. |
| YOU | you | 18 clips. Pointing/indexical sign. |
| HE | he | 17 clips. Pointing sign. Paired with SHE as the primary ambiguity candidate — see open decision 1. |
| SHE | she | 17 clips. Pointing sign differing from HE mainly in referent, not handshape — the strongest available candidate for demo scene 2's genuine confusion pair. |
| HOSPITAL | hospital | 21 clips. Core noun anchoring the scenario. |
| HELP | help | 16 clips — joint-lowest count in the active set. High-value verb for this context. |
| GO | go | 17 clips. Directional verb. |
| SIT | sit | 19 clips. "Please sit" is a natural reception-desk instruction and gives the Speech→Sign direction something concrete to render. |
| OKAY | okay | 23 clips. Acknowledgment token; also the natural clean-follow-up segment that resumes the clarification ladder in FR-11. |
| WHAT | what | 17 clips. ISL commonly places question words at the end of the clause — this constrains T1.8's gloss ordering, not just recognition. |
| WHERE | where | 17 clips. Same clause-position note as WHAT. |
| TODAY | today | 17 clips. ISL typically establishes time at the start of a clause — also relevant to T1.8's ordering. |

**Total: 16 classes, 326 clips, min 16 per class.**

---

## Phase 2 vocabulary — deferred, not rejected

Adding a class is a cheap retrain of the same small classifier head (T1.4), so
none of this is structurally blocked. Two different reasons for deferral, and
the distinction matters:

**(a) Dropped for insufficient data — the corpus has these, but too few clips.**
Training and holding out a test set on 2–6 examples cannot produce a calibrated
confidence, and FR-2 forbids a placeholder. Recording additional takes would
promote these cheaply, since the class already exists upstream.

| gloss_id | clips available | note |
|---|---|---|
| WHEN | 2 | Question word; its absence leaves WHAT and WHERE covering the question-word role. |
| ME | 3 | First-person indexical. Its absence is the most keenly felt gap — demo sentences must be built around YOU rather than ME. |
| COME | 4 | Would have paired with GO as a directional ambiguity candidate. |
| SORRY | 6 | Useful opener for an unscheduled walk-up. |

**(b) Not in the corpus at all — require original recording.**
These are the hospital-specific core. Every one needs 8–10 self-recorded takes
(more than the 3–5 originally planned, to offset the class imbalance against
the ~17-clip corpus classes).

`DOCTOR`, `NURSE`, `APPOINTMENT`, `MEDICINE`, `PAIN`, `WANT`, `NEED`, `MEET`,
`WAIT`, `HAVE`, `WHO`, `TOMORROW`, `NAME`, `UNDERSTAND`

**Honest consequence to state plainly:** with `DOCTOR`, `MEDICINE`, `PAIN` and
`APPOINTMENT` all deferred, Stage 1 does not yet cover the medical vocabulary
its own scenario is built around. `HOSPITAL` and `HELP` carry the context alone.
This is a disclosed limitation of the current build, not something to paper over
in a demo narrative.

## Also available free in the same corpus, if wanted

24 further glosses ship with the dataset at no recording cost. Those with ≥15
clips: `friend`(38), `school`(37), `market`(36), `okay`(23), `sit`(19),
`eat`(18), `mother`(18), `water`(18), `food`(17), `she`(17), `he`(17),
`sister`(17), `student`(17), `tea`(17), `teacher`(17), `drink`(16),
`father`(16). Thin ones to avoid for the same reason as group (a) above:
`home`(1), `stand`(2), `brother`(3), `goodbye`(3), `read`(3), `write`(3),
`stop`(4).

---

## Entries deliberately excluded, and why

Not oversights — each protects a specific demo scene.

| Excluded term | Why it stays out |
|---|---|
| Any personal name | Must fall through to fingerspelling to satisfy FR-7 and demo scene 4. A name sign would destroy the refusal scene. |
| Any specific drug name | Same — the intended out-of-vocabulary trigger for the `FINGERSPELLING` / `UNMATCHED` coverage path in T1.9. |
| Numbers | ISL number handshapes are a system, not 10 isolated signs. No Stage 1 demo scene requires one. |
| Fingerspelled alphabet handshapes | Supplied by `spoken-to-signed-translation` as built-in behavior (architecture v3 §6.8) — but see the blocker below. |

## ⚠️ Blocker affecting demo scene 4 — verified 2026-09-15

`spoken-to-signed-translation` ships fingerspelling lexicons for `ase, asq,
bzs, cse, csq, eso, gsg, gss, ise, jos, lls, mfs, psr, sgg, ssp, svk, swl,
tsm, ukl`. **`ins` (Indian Sign Language) is not among them** — `ise` is
*Italian*, not Indian. There is therefore **no ISL fingerspelling data
bundled**, so an out-of-vocabulary term will currently resolve `UNMATCHED`
rather than `FINGERSPELLING`.

**RESOLVED 2026-09-15 (T1.9): option 3 taken — out-of-vocabulary terms resolve
`UNMATCHED`.** A real ISL manual alphabet *was* located —
[`Hemg/Indian_sign_language_dataset`](https://huggingface.co/datasets/Hemg/Indian_sign_language_dataset),
42,745 images over A–Z plus digits 1–9, and visually confirmed to be genuine
**two-handed** ISL rather than a mislabelled ASL set. It was rejected as a
lexicon source for a concrete reason: the images are 128×128 crops of hands
alone, with no torso or arms. Every other pose in this project carries
`POSE_LANDMARKS`, which is what lets the avatar place hands in signing space;
manufacturing a body for these would mean inventing data the source does not
contain — the exact fabrication the project exists to refuse. MediaPipe
detection on the crops is also inconsistent, finding only one of two hands on
several letters.

Recorded for a later phase: the dataset is a viable fingerspelling source for
anyone willing to do the body-placement work properly, which is a piece of real
engineering rather than a config change.

Demo scene 4 is described in architecture v3 §10 as "the strongest 25 seconds."
Three options, considered before T1.9:

1. Build a minimal ISL manual-alphabet lexicon. ISL uses a **two-handed**
   alphabet, structurally unlike ASL's one-handed one, so ASL poses are not a
   silent substitute.
2. Fall back to ASL (`ase`) fingerspelling and **say so explicitly** in the
   demo and the decision log — a disclosed substitution, never an unmarked one.
3. Let the scene resolve `UNMATCHED` and present refusal itself as the outcome.
   This is arguably the most honest option and still demonstrates
   refuse-to-fabricate (FR-12), just without a fallback rendering.

Verified alongside this: `--signed-language` is **not** validated against an
ISO allowlist (`gloss_to_pose/languages.py` holds only a two-entry backup map),
so `ins` is safe to use as a free-form selector. This closes the question T1.9
flagged as unverified.

**Checked 2026-09-15 and ruled out as a fix:** the ISLRTC dictionary below does
*not* contain the manual alphabet. It has signs for the words `Fingerspell`,
`Alphabet`, `Greek_Alphabets` and `Ancient_Alphabets`, but no individual A–Z
letter entries (0 of 26 found), and only `Zero` of the digits. An ISL
fingerspelling lexicon would still have to be recorded or sourced elsewhere.

---

## Vocabulary expansion path — ISLRTC / data.gov.in dictionary

[`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`](https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded)
— the official ISLRTC dictionary, **13,665 clips, MIT licensed**, professional
studio recording on a plain background with the full signing space in frame.
Verified 2026-09-15.

Coverage: **16/16** of the active vocabulary above, and **12/18** of the
deferred words — including `DOCTOR`, `NURSE`, `APPOINTMENT`, `MEDICINE`,
`PAIN`, `MEET`, `WAIT`, `HAVE`, `TOMORROW`, `NAME`, `COME`, `SORRY`. Missing:
`WANT`, `NEED`, `WHO`, `UNDERSTAND`, `ME`, `WHEN`.

Measured cost on the target M5: **1.7s per clip, 88KB per `.pose`** → the whole
dictionary is **~6.3 hours of extraction and ~1.21GB** of pose data. Feasible,
but it does not belong in a normal git repo — commit the extraction script and
generate locally, or use Git LFS.

**Its usefulness is asymmetric, and that asymmetry should be stated plainly
rather than blurred:**

- **Speech→Sign (avatar output): scales fully.** The gloss→pose lexicon is a
  *lookup* needing exactly one reference pose per gloss, which is precisely
  what a dictionary provides and precisely the `index.csv` format described
  above. A five-figure avatar vocabulary is genuinely reachable.
- **Sign→Speech (recognition): does not scale.** One clip per word cannot train
  a classifier, and DTW over thousands of single templates from one signer
  would collapse on visually similar signs. Recognition stays a small curated
  set — which is exactly what architecture v3 §11.1 already says, and what
  FR-1 requires.

The honest framing this supports: *Gestura signs a large ISL vocabulary to the
Deaf user and recognizes a smaller validated set back.* That is a stronger and
more defensible claim than implying both directions scale equally.

---

## Open decisions for the owner

1. ~~**Confirm the ambiguity pair empirically.**~~ **RESOLVED 2026-09-15 by
   T1.4's held-out run — the pair is `HE`/`SHE`.** The classifier confused them
   in both directions on real held-out data (`SHE`→`HE` twice, `HE`→`SHE` once),
   at confidences of 0.077, 0.040 and 0.128 — all well inside the clarification
   band. Demo scene 2's low confidence is therefore genuine and reproducible,
   not staged, satisfying FR-2. `YES`/`NO` is a confirmed secondary pair
   (`YES`→`NO` twice, `NO`→`YES` once) if a backup beat is wanted.
2. **Resolve the fingerspelling blocker** above before T1.9.
3. **Confirm the articulation notes.** Per the honesty note, the `notes` column
   is unvalidated.
4. **Decide on `PLEASE`.** If it is not naturally signed in ISL here, drop it
   rather than keep an artificial entry.

## Extraction checklist (feeds T1.3)

No recording is required for the active 16 — the pipeline runs on downloaded
clips. For Phase 2 group (b), when recording does happen:

- One subdirectory per `gloss_id` under `data/vocab/raw_video/`.
- **8–10 takes** per sign, varying speed slightly, to offset imbalance against
  the corpus classes' ~17 clips.
- Fixed camera, consistent lighting, upper body and both hands fully in frame
  for the entire sign.
- Also record one clean take of the chosen out-of-vocabulary name, so the
  refusal scene can be rehearsed end to end.
