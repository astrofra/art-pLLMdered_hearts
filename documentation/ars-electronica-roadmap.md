# Ars Electronica Roadmap

## Scope

This roadmap addresses the main weaknesses identified by the Ars Electronica reviewers across the two existing states of the project:

- **Prototype 1**: the automated LLM-versus-parser experiment in `pLLuMdered-hearts_old`
- **Final version**: the mediation-oriented installation in `/Users/fra/dev/pLLMdered_hearts`

The goal is not only to improve the artwork itself, but also to make its research claims, visitor experience, and documentation legible for a future submission.

## Reviewer Critiques Translated Into Workstreams

| Reviewer concern | Practical response |
| --- | --- |
| The paper wavers between a quasi-empirical experiment and a conceptual essay. | Separate the two phases more clearly and define distinct claims for each. |
| The metrics and charts look stronger than the protocol actually allows. | Reframe Prototype 1 as illustrative unless the experiment is strengthened with repeated runs, baselines, and clearer limits. |
| The mediation phase is the real contribution, but it is under-described. | Expand the final installation documentation: segmentation method, prompting strategy, semantic matching, and concrete examples. |
| It is not clear enough what visitors actually see and how the installation works in exhibition. | Produce a visitor-centered description of the final installation, with screens, timing, language, and sample sequences. |
| French commentary over English gameplay creates accessibility issues. | Add an English-facing mode, or bilingual output, for documentation and exhibition contexts. |
| Structure, references, and formatting are inconsistent. | Prepare a clean publication package with a standard section structure and consistent references. |

## Roadmap Overview

### Phase 1. Reposition the project claims

**Objective:** make each version of the project say one thing clearly.

#### Prototype 1

- Reframe Prototype 1 as a **demonstration of incompatibility** between parser ontology and probabilistic language generation, not as an LLM benchmark.
- State explicitly that the charts are **illustrative traces of failure modes**, unless the protocol is expanded.
- Turn the "ribbon problem" into the central case study for this phase.

#### Final version

- Define the final installation as the **primary artistic contribution**.
- Shift the core claim from "can an LLM play?" to "how can an LLM mediate an obsolete interactive work?"
- Present the walkthrough-based system as a deliberate curatorial choice, not a fallback.

#### Deliverables

- A one-page project statement distinguishing the two phases.
- A revised abstract and a short "research claims" note for future calls.

### Phase 2. Consolidate Prototype 1 as a rigorous negative result

**Objective:** either strengthen the experiment or deliberately reduce its epistemic ambition.

#### Must-have actions

- Run repeated sessions for each history-window setting instead of relying on isolated runs.
- Test at least one comparison model or one non-LLM baseline.
- Record a fixed experimental matrix: model, temperature, prompt version, history length, stop condition, date, and hardware context.
- Distinguish clearly between:
  - parser rejection
  - semantic drift
  - command repetition
  - retry-after-failure
  - successful state-changing actions
- Build a compact failure taxonomy from the logs:
  - lexical misrecognition
  - invented affordances
  - loop persistence
  - over-interpretation of descriptive details

#### Recommended actions

- Create one reproducible benchmark script that regenerates the full log corpus and figures.
- Add a short baseline using hand-authored parser-safe heuristics or walkthrough snippets, not to "beat" the model but to show where failure comes from.
- Produce a small appendix of emblematic sequences: `ribbon`, repeated object fixation, navigation loops, false repair strategies.

#### Expected outcome

Prototype 1 becomes either:

- a modest but defensible experimental section, or
- a sharply framed failure study that supports the conceptual argument without pretending to prove more than it can.

### Phase 3. Deepen the final installation as a mediation system

**Objective:** make the second phase concrete, analyzable, and clearly stronger than the first.

#### Archive preparation and segmentation

- Document exactly how Amy Briggs' interview is segmented:
  - source file
  - cleaning steps
  - segmentation criteria
  - average segment length
  - naming convention
- Explain which transformations are deterministic and which are AI-mediated.
- Preserve one frozen corpus version for publication and one working corpus for ongoing iteration.

#### Prompting and commentary design

- Document the exact commentary prompt used by the final system.
- Define the intended voice of the mediator: historical, interpretive, speculative, ironic, or mixed.
- Add prompt variants and compare their effects on tone and relevance.
- Collect at least 8 to 12 representative commentary examples, including successful and weak ones.

#### Semantic matching and selection logic

- Document the embedding model, similarity calculation, cooldown rules, repetition filters, and queue logic.
- Explain why a selected video excerpt was chosen in at least a few concrete cases.
- Add a lightweight trace output for each step:
  - current game situation
  - generated commentary
  - selected interview segment
  - similarity score
  - rejected recent candidates if applicable

#### Visitor experience

- Write a visitor-facing sequence description from entry to exit:
  - what appears on screen 1
  - what appears on screen 2
  - how often commentary appears
  - how video clips are triggered
  - what happens when nothing is selected
- Clarify the dramaturgy of the installation:
  - why the game follows a walkthrough
  - why the AI comments instead of acting
  - how the two narrative voices intersect
- Produce a short exhibition script with screenshots and timestamps.

#### Language and accessibility

- Add an English documentation mode for the final installation.
- Decide between three options for exhibition:
  - English commentary only
  - French commentary with English subtitles
  - bilingual dual-caption mode
- Use the same language policy in the paper, the screenshots, and the exhibition documentation.

#### Expected outcome

The final version becomes legible as a complete mediation apparatus rather than a brief conceptual pivot after the failed gameplay experiment.

### Phase 4. Build evidence from the final installation, not only from the prototype

**Objective:** show what the mediator actually achieves.

#### Actions

- Curate 4 to 6 strong "emulation encounter" case studies.
- For each case, document:
  - game passage
  - LLM commentary
  - matched interview segment
  - why the association is interesting
  - whether it is historically grounded, poetically surprising, or both
- Include at least one counter-example where the mediation is weak, generic, or misleading.
- If possible, gather light exhibition evidence:
  - visitor observations
  - photos of the setup
  - short notes on reading time, attention flow, or confusion points

#### Expected outcome

Future readers are not only told that the LLM becomes a mediator; they are shown how that mediation operates, where it succeeds, and where it remains unstable.

### Phase 5. Prepare a cleaner publication package

**Objective:** align the project documentation with conference expectations without flattening its artistic identity.

#### Article structure

- Rebuild the paper with a conventional spine:
  - Introduction
  - Theoretical context / related work
  - Prototype 1: parser-LLM friction
  - Final installation: semantic mediation
  - Discussion
  - Conclusion
- Reduce the space devoted to charts unless the experiment is significantly strengthened.
- Give more space to the installation logic, visitor experience, and case studies from the final version.

#### Figures and appendices

- Keep one strong figure for Prototype 1.
- Add at least two stronger figures for the final version:
  - mediation pipeline
  - annotated visitor view or screen choreography
- Move raw protocol details, repeated charts, or extra logs to an appendix or repository documentation.

#### Editorial cleanup

- Fix ACM or target venue formatting from the start.
- Harmonize bibliography style and incomplete references.
- Standardize terminology: parser, ontology, mediator, post-executable readability, semantic intertextuality.

#### Expected outcome

The submission will read as a focused paper about mediated readability, supported by a preliminary failure study, instead of as two competing papers sharing the same title.

## Suggested Timeline

### Short term: 2 to 3 weeks

- Reposition the claims.
- Freeze the distinction between Prototype 1 and the final installation.
- Write the missing documentation for segmentation, prompting, and visitor experience.
- Decide on an English or bilingual presentation strategy.

### Mid term: 3 to 5 weeks

- Re-run Prototype 1 with a controlled protocol.
- Produce the failure taxonomy and revised figures.
- Collect final-installation case studies with traceable examples.

### Pre-submission: 2 weeks

- Rewrite the paper around the final installation.
- Reduce or qualify experimental claims where needed.
- Clean formatting, references, captions, and figure balance.

## Priority Order

If time is limited, the priority should be:

1. **Clarify the project's main claim**: the final installation is the core contribution.
2. **Make the mediation phase explicit**: segmentation, prompts, matching, visitor experience, case studies.
3. **Either strengthen or downgrade Prototype 1's empirical ambition**.
4. **Resolve language accessibility and publication formatting issues**.

## Definition of Success

This roadmap will have worked if a future reviewer can immediately understand:

- why Prototype 1 fails,
- why that failure matters,
- what the final installation does instead,
- what a visitor concretely experiences,
- and why the second phase is the actual artistic and scholarly contribution.
