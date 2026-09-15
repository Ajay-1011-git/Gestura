# Words that must not be fingerspelled

Curated by hand. Every word here is an ordinary English word with a real,
established ISL sign. If one of them fails to resolve, that is a **gap in this
deployment's 16-sign render lexicon** (PRD R-10) — not a word that needs
spelling out.

## Why this list exists

FR-27, read literally, says any term with no lexicon match and no
language-backup match should be fingerspelled before falling back to
`UNMATCHED`. Applied to a 16-sign lexicon, that produces nonsense: "I need an
ambulance" came out as `I` + `NEED` + `AMBULANCE` all spelled letter by letter,
fifteen handshapes for a three-word sentence.

That is not what fingerspelling is for. Interpreters fingerspell proper nouns,
acronyms, and terms with genuinely no sign. Nobody spells "I-N-E-E-D" — a Deaf
viewer would read it as noise, and presenting it as a translation is a new way
of producing confident-looking bad output, which is the one thing this project
is built not to do.

So the rule is narrowed to FR-27's evident intent, and the narrowing is written
down rather than buried in a heuristic: **spell what warrants spelling, refuse
the rest honestly.** A word on this list that has no pose surfaces as
`UNMATCHED` and the waterfall refuses — the same visible, logged refusal Stage 1
gave, which correctly reports a lexicon gap instead of disguising it as a
successful spelling.

Remove a word from this list only when it is genuinely spelled in ISL. Adding
the missing signs to the render lexicon is the real fix, and it is vocabulary
work deliberately out of Stage 2's scope.

## The list

Function words — pronouns, articles, prepositions, conjunctions, auxiliaries.
All have ISL forms; none is ever spelled.

| Word |
|---|
| a |
| an |
| the |
| i |
| me |
| my |
| mine |
| you |
| your |
| yours |
| he |
| him |
| his |
| she |
| her |
| hers |
| it |
| we |
| us |
| our |
| they |
| them |
| their |
| this |
| that |
| these |
| those |
| is |
| am |
| are |
| was |
| were |
| be |
| been |
| do |
| does |
| did |
| have |
| has |
| had |
| will |
| would |
| can |
| could |
| shall |
| should |
| may |
| might |
| must |
| not |
| no |
| yes |
| and |
| or |
| but |
| if |
| then |
| than |
| because |
| so |
| for |
| to |
| of |
| in |
| on |
| at |
| by |
| with |
| from |
| about |
| into |
| over |
| under |
| up |
| down |
| out |
| off |
| again |
| here |
| there |
| now |
| very |
| more |
| most |
| some |
| any |
| all |
| one |

Common verbs and content words with established ISL signs.

| Word |
|---|
| need |
| want |
| go |
| come |
| give |
| take |
| make |
| get |
| see |
| look |
| know |
| think |
| say |
| tell |
| ask |
| answer |
| help |
| work |
| live |
| eat |
| drink |
| sleep |
| sit |
| stand |
| walk |
| run |
| wait |
| stop |
| start |
| finish |
| open |
| close |
| like |
| love |
| feel |
| good |
| bad |
| big |
| small |
| new |
| old |
| hot |
| cold |
| happy |
| sad |
| sorry |
| please |
| thanks |
| hello |
| goodbye |
| okay |
| name |
| time |
| day |
| today |
| tomorrow |
| yesterday |
| morning |
| night |
| year |
| week |
| month |
| home |
| house |
| school |
| water |
| food |
| money |
| friend |
| family |
| mother |
| father |
| child |
| man |
| woman |
| people |
| where |
| what |
| when |
| who |
| why |
| how |
| which |

## The 40-word recognition vocabulary

Every gloss below has recorded ISL video in `data/vocab/` — the corpus exists
precisely because these are real signs. The render lexicon covers only 16 of
them, so the other 24 miss on lookup. That miss is the 40-vs-16 asymmetry named
in R-10, and the honest response to it is a refusal that reports a missing sign,
not a spelling that hides one.

| Word |
|---|
| brother |
| come |
| drink |
| eat |
| father |
| food |
| friend |
| go |
| goodbye |
| he |
| hello |
| help |
| home |
| hospital |
| market |
| me |
| mother |
| no |
| okay |
| please |
| read |
| school |
| she |
| sister |
| sit |
| sorry |
| stand |
| stop |
| student |
| tea |
| teacher |
| thank you |
| today |
| water |
| what |
| when |
| where |
| write |
| yes |
| you |
