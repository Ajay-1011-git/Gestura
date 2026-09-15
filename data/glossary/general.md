# Domain glossary — `general`

The default domain, selected when no room context is supplied. It holds only
terms whose ISL gloss is stable across every setting Setu might be deployed in
— greetings, courtesies, the question words, and the handful of everyday verbs
the Stage 1 lexicon actually renders.

**Deliberately the smallest of the three.** A general glossary that tried to
cover everything would be a worse version of the LLM reasoning step it sits in
front of, and every wrong entry here costs more than a miss would: a miss falls
through and gets reasoned about properly, while a wrong hit is never revisited.
Anything whose correct gloss depends on the room belongs in `medical.md` or
`technical.md`, not here.

**How to read the table.** `Term` is the spoken-English surface form matched
exactly (case- and whitespace-insensitively). `Gloss` is the ISL gloss token
emitted on a hit. `Note` records why the entry exists or what it is *not*.

A gloss here is not a promise that the sign renders — `data/lexicon/` covers 16
signs, and a gloss outside that set still falls through to fingerspelling or a
refusal downstream. The glossary's job is getting the *right* gloss, not
guaranteeing a pose exists for it.

| Term | Gloss | Note |
|---|---|---|
| hello | HELLO | In the Stage 1 lexicon. |
| hi | HELLO | Same sign; separate surface form so the exact match lands. |
| thank you | THANK-YOU | Hyphenated: one sign, two English words. |
| thanks | THANK-YOU | |
| please | PLEASE | In the Stage 1 lexicon. |
| sorry | SORRY | Extracted vocabulary exists; not in the 16-sign render lexicon. |
| yes | YES | In the Stage 1 lexicon. |
| no | NO | In the Stage 1 lexicon. |
| okay | OKAY | In the Stage 1 lexicon. Also the most common filler — T2.5 suppresses the repeated form before it ever reaches here. |
| help | HELP | In the Stage 1 lexicon. |
| where | WHERE | ISL places question words last; the reasoning step handles ordering, not this table. |
| what | WHAT | In the Stage 1 lexicon. |
| when | WHEN | Extracted vocabulary exists. |
| who | WHO | |
| today | TODAY | In the Stage 1 lexicon. Time markers lead the clause in ISL. |
| tomorrow | TOMORROW | |
| yesterday | YESTERDAY | |
| go | GO | In the Stage 1 lexicon. |
| sit | SIT | In the Stage 1 lexicon. |
| wait | WAIT | Distinct from the safe-mode hold message, which is spoken English, not signed. |
| you | YOU | In the Stage 1 lexicon. |
| he | HE | In the Stage 1 lexicon. |
| she | SHE | In the Stage 1 lexicon. |
| name | NAME | The sign that most often precedes a fingerspelled proper noun (T2.6). |
