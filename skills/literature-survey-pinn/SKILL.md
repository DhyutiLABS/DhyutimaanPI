---
name: literature-survey-pinn
description: Conduct a focused literature survey on Physics-Informed Neural Network (PINN) topics and produce a structured knowledge base that the pinn-problem-spec skill can consume. Use this skill whenever the user asks for a literature survey, lit review, "state of the art", or "recent work" related to PINNs, physics-informed machine learning, scientific machine learning (SciML), neural operators (DeepONet, FNO), or specific PINN variants (causal PINN, XPINN, PINNsFormer, variational PINN, etc.). Also trigger when the user uploads PINN papers and asks for synthesis, or asks "what's been tried" for a specific PDE or failure mode. Produces a structured markdown knowledge base with a PINN-specific taxonomy and a "what to try next" section that feeds into problem framing.
---

# PINN Literature Survey

This skill conducts a focused, structured literature survey on PINN topics and emits a knowledge base in a format the `pinn-problem-spec` skill can directly consume. The output is *operational*, not encyclopedic — the question this survey answers is always "what does this tell us to try next?", not "everything ever written on PINNs."

This skill is a specialization of the general `literature-survey` skill (if present in the environment). The PINN specialization adds:

1. A taxonomy that maps every paper into known PINN variants and failure modes
2. A "PDE-specific" view that surfaces what's been tried for the user's PDE class
3. A handoff format that the problem-spec skill reads directly

## When to use this skill

Trigger when the user:

- Asks for a literature survey, review, or summary on PINNs or any PINN variant
- Mentions a specific PDE and asks "what's been done with PINNs for this?"
- Uploads PINN papers and asks for synthesis
- Asks about a failure mode ("why are PINNs bad at stiff problems?") and wants references
- Is preparing to write a problem spec and needs grounding in prior work

Do *not* trigger this skill for implementation questions, analysis of completed runs, or general ML literature searches not connected to physics-informed methods.

## Sources to search

Search in this order of priority:

1. **arXiv** (cs.LG, physics.comp-ph, math.NA) — most PINN work is here first
2. **Semantic Scholar** — for citation graphs and related work
3. **OpenReview** (ICLR, NeurIPS) — for peer-reviewed versions and reviews
4. **Journal of Computational Physics** and **Computer Methods in Applied Mechanics and Engineering** — for the engineering side
5. **GitHub** — search for high-star implementations; these often reveal which methods actually work in practice
6. **Author group pages** — Karniadakis (Brown), Perdikaris (UPenn), Em Karniadakis collaborators, Steve Brunton (UW), Nathan Kutz (UW), the DeepXDE / Modulus maintainers

If MCP connectors for any of these are available, prefer them over web search — they return structured metadata.

## Query expansion

A good PINN survey query is rarely a single phrase. Expand the user's query along multiple axes before searching:

- **Variants**: vanilla PINN, XPINN (extended), cPINN (conservative), causal PINN, variational PINN, PINNsFormer, FBPINN, finite-basis PINN, hard-constraint PINN
- **Architectures**: MLP, SIREN, Fourier features, Gaussian process baselines, neural operators (DeepONet, FNO, GNO), Transformer-based
- **Training tricks**: NTK-based weighting, GradNorm, ReLoBRaLo, soft attention masks, causal weighting (Wang et al. 2022), curriculum learning, two-stage Adam+L-BFGS
- **Failure modes**: spectral bias, loss imbalance, propagation failure, gradient pathologies, multi-scale problems, stiff problems
- **PDEs**: the specific equation the user cares about (Burgers', wave, Navier-Stokes, Allen-Cahn, Helmholtz, Schrödinger, ...)

For each axis, generate 2–3 search queries. The PINN literature uses inconsistent terminology, so synonym variation matters.

## Output: PINN knowledge base

Always produce a single markdown file named `pinn-knowledge-base.md` with this structure:

```markdown
# PINN Literature Survey: <topic>

## Scope
What we searched for, what's in/out of scope, when this survey was done.

## Headline findings
3–5 bullets. The most decision-relevant takeaways. What works, what's contested, what's open.

## Taxonomy of relevant work
Group papers by variant category. For each category, give:
- 1-paragraph description of the variant
- Key papers (with year and venue)
- What problem it addresses
- Known limitations or open critiques

Categories to consider (only include those represented in the literature you found):
- Vanilla PINNs (Raissi et al. 2019 baseline)
- Loss weighting schemes
- Architectural variants (Fourier features, SIREN, etc.)
- Domain decomposition (XPINN, FBPINN, cPINN)
- Causal and time-marching variants
- Neural operators (DeepONet, FNO) — note these are PINN-adjacent, not PINN
- Hard-constraint methods (output transformation, optimization-on-manifold)
- Variational and weak-form approaches

## Findings by PDE class
If the user has a specific PDE in focus, dedicate a section to it:
- What's been tried
- What worked, what didn't
- Reference benchmarks (if any)
- Known failure modes specific to this PDE

## Common failure modes documented in the literature
For each failure mode, cite the paper(s) that documented it and any proposed remedies.

## Open problems and contested points
Where the literature disagrees, where benchmarks are lacking, where evaluation protocols differ.

## Implementation landscape
- Libraries: DeepXDE, NVIDIA Modulus, NeuroDiffEq, IDRLnet
- Notable open-source implementations (with stars/activity)
- Which papers ship code, which don't

## Recommended starting points for problem-spec
3–5 specific suggestions, each phrased so the problem-spec skill can act on it.
Example: "For the 1D wave equation with sharp ICs, causal weighting (Wang, Sankaran, Perdikaris 2022) is the strongest baseline; vanilla PINNs fail with error growing in t."

## Sources
Numbered list with full citations and links/DOIs.
```

## Workflow

### Step 1: Establish scope

Read the user's request and the conversation context. Identify:

- Is there a specific PDE or problem class?
- Is there a specific failure mode in focus?
- Is the user looking for the "state of the field" or a targeted question?
- Are there papers already uploaded that should be included verbatim?

If the scope is broad ("survey PINNs"), narrow with the user before searching — a broad survey is rarely what they need and is hard to make operational.

### Step 2: Expand queries and search

Generate 6–12 search queries across the axes listed above. Run them in parallel where possible. For each result:

- Capture title, authors, year, venue, link
- Extract: what variant, what PDE, what was the claim, what was the evidence
- Note whether code is available

Do not stop at the first page of results — PINN-relevant work is sometimes 2–3 pages deep, especially on Semantic Scholar.

### Step 3: Read the most important papers carefully

After the broad scan, pick 5–10 papers for closer reading based on:

- Citation impact (but discount this for recent papers)
- Relevance to the user's specific PDE or failure mode
- Methodological novelty
- Availability of code (papers with code are more useful as references)

For each, extract:

- Governing equation studied
- Architecture and training recipe
- Reference / baseline used
- Headline numerical result (with caveats)
- Stated limitations

### Step 4: Build the taxonomy

Don't pretend every paper fits cleanly. Some belong in multiple categories; some are outliers. Be honest in the categorization, and flag papers that are over-cited or under-tested.

### Step 5: Surface contested points

The PINN literature has real disagreements — e.g., whether NTK-based weighting helps, how to evaluate fairly, whether neural operators should be compared to PINNs at all. Include a "contested" section that names these debates and points to papers on each side. This is where survey adds value over a paper list.

### Step 6: Write the recommendations section

This is the handoff to problem-spec. Each recommendation should be specific enough that the problem-spec skill can use it as a starting point. Cite the source.

### Step 7: Be honest about gaps

If the literature is thin on the user's specific question, say so. Don't pad the survey with weakly-related papers. A short, honest survey beats a long one that overstates its base.

## Citation discipline

Every claim about what a paper showed must be tied to that paper, by author and year at minimum. Do not paraphrase results from memory — if you can't find the source, leave the claim out. PINN literature contains a lot of folklore ("PINNs can't do turbulence", "Fourier features always help") that is more cited than supported; flag folklore explicitly when you find it.

When searching the web, follow the citation rules in the environment's search instructions — paraphrase, limit quotes, attribute claims.

## Anti-patterns

- **Producing an encyclopedic list with no narrative.** The user wants to know what to try, not what exists.
- **Citing recent arXiv papers as established results.** Note the venue and review status.
- **Conflating PINNs and neural operators.** They solve different problems; treat them as separate categories.
- **Reproducing benchmark numbers without their caveats.** Papers often use favorable evaluation setups; flag this.
- **Hand-waving on what "doesn't work."** If a paper claims a method fails on a problem, cite it; otherwise it's folklore.
- **Pretending the literature is more settled than it is.** Disagreements are signal.

## Handoff

After the knowledge base is written, summarize for the user in 5–8 lines: scope, top 3 takeaways, and the 2–3 recommendations most relevant to their next problem-spec. Then say:

> Knowledge base is at `pinn-knowledge-base.md`. The "Recommended starting points for problem-spec" section is the natural input to the problem-spec skill when you're ready.
