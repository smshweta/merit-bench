# When Does Memory Help? A Cost-Aware Evaluation of Long-Term Memory in Tool-Using LLM Agents

**Shweta Mishra**
*(Independent Research / affiliation at time of submission)*

> **STATUS: DRAFT v0.1 — Sections 1–4 are complete prose. Sections 5–6 are templates to be filled with real experimental results. Do NOT submit before running experiments. Citations marked [VERIFY] must be checked against the actual papers before submission.**

---

## Abstract

Long-term memory is widely assumed to improve the performance of LLM-based agents, and a rapidly growing ecosystem of memory systems — retrieval-augmented stores, rolling summarization, and structured fact memories — now competes on conversational recall benchmarks such as LoCoMo and LongMemEval. However, these benchmarks measure *question answering over dialogue history*, not the setting in which memory is most consequential in production: **tool-using agents executing multi-step tasks across sessions**. In this work we present **MERIT** (Memory Evaluation for Realistic Instrumented Tasks), a benchmark and evaluation harness that measures the *marginal utility* of memory for task-executing agents under explicit cost accounting. MERIT extends a tool-calling agent environment with (i) cross-episode task dependencies, where facts established in earlier episodes are required for later task success; (ii) controlled **memory corruption**, where stale or incorrect memories are injected to measure harm; and (iii) full token, dollar, and latency accounting for every memory operation. We evaluate N memory architectures (none, full-context replay, vector RAG, rolling summary, structured fact store, hybrid) across M models. We introduce three metrics absent from prior evaluations: **Memory Utilization Rate** (whether retrieved memories are actually used in the agent's decisions), **Ignore Rate**, and **Stale-Memory Harm** (task-success degradation under corrupted memory). Our results show that **[RESULT PLACEHOLDER: e.g., "memory improves task success by X% on dependency tasks but degrades success by Y% on independent tasks due to distraction, and no system pays for itself below a task-value threshold of $Z"]**. We release MERIT and all harness code to enable reproducible, cost-aware comparison of agent memory systems.

**Keywords:** LLM agents, long-term memory, benchmark, evaluation, tool use, cost analysis

---

## 1. Introduction

LLM-based agents that plan, call tools, and act over multiple sessions are moving rapidly into production — customer support, IT operations, coding, and commerce. A central architectural question for every such deployment is whether and how to give the agent *long-term memory*: information persisted across sessions and selectively recalled. A vibrant ecosystem of memory systems has emerged (Mem0, Letta/MemGPT-style OS memories, Zep, LangGraph checkpointing, bespoke RAG stores), and vendors compete on recall benchmarks.

Yet the evaluation of agent memory has three blind spots:

**Blind spot 1: Conversational QA is not task execution.** The dominant benchmarks — LoCoMo, LongMemEval, and BEAM — test whether a system can answer questions about long, multi-session *conversations*. Production agents, by contrast, must use remembered facts to *choose the correct tool call*: the right customer ID, the previously agreed refund policy exception, the deployment that caused last week's incident. Whether conversational recall scores transfer to task-execution utility is an open empirical question. [Expand: 1–2 sentences citing that LoCoMo/LongMemEval are explicitly conversational QA; cite Maharana et al. 2024, Wu et al. 2025.]

**Blind spot 2: Memory is assumed to help; harm is unmeasured.** Recent surveys note that agents sometimes ignore retrieved memory records or use them inconsistently, and benchmarks like HaluMem have begun to examine memory hallucination and consistency — but in conversational settings. In task execution, a stale memory (a customer's *old* address, a rolled-back configuration) is not just unhelpful; it produces a *confidently wrong action*. No existing benchmark quantifies this failure mode under controlled corruption.

**Blind spot 3: Cost is reported, but marginal utility is not.** Memory systems report token counts alongside accuracy, but the decision practitioners face is economic: does adding memory raise task success enough to justify its per-task cost (extra retrieval calls, larger prompts, write-time summarization)? We are not aware of any evaluation that reports *cost-adjusted marginal utility* — the change in success probability per marginal dollar — across memory architectures.

We address all three with **MERIT**, a benchmark and open-source harness for cost-aware evaluation of memory in tool-using agents. Our contributions:

1. **A task suite with controllable memory dependency.** We construct episodic tool-use tasks in three domains (customer support, IT operations, personal assistant) where a tunable fraction of episodes depend on facts established in earlier episodes. This lets us measure memory utility as a function of *actual dependency*, separating genuine recall value from placebo effects.
2. **Controlled memory corruption.** We inject stale, contradictory, and irrelevant memories at known rates and measure Stale-Memory Harm — the drop in task success attributable to corrupted memory — along with whether systems abstain or verify before acting.
3. **Three new metrics.** Memory Utilization Rate (MUR), Ignore Rate, and Cost-Adjusted Marginal Utility (CAMU), defined in §3.4, computed automatically by the harness.
4. **An empirical study** of [N] memory architectures × [M] models ([list]), totaling [K] agent episodes, with full statistical treatment (paired bootstrap over task instances, [S] seeds).
5. **Open-source release** of the benchmark, harness, and all traces.

## 2. Related Work

### 2.1 Memory systems for LLM agents
[Write 2 paragraphs. Cover: MemGPT/Letta (OS-style paging), Mem0 (structured fact extraction; ECAI 2025), Zep/Graphiti (temporal knowledge graphs), rolling summarization (LangChain/LangGraph patterns), A-Mem, ENGRAM, Hippocampus (2025–2026 arXiv systems). Framing: systems papers propose architectures and evaluate on conversational QA; none evaluate task-execution utility under cost. Cite each system's own reported benchmark to show the pattern.]

### 2.2 Benchmarks for agent memory
[Write 2 paragraphs. LoCoMo (Maharana et al., 2024): 1,540 QA over multi-session dialogues (~600 turns, ~16K tokens; single-hop/multi-hop/temporal/open-domain). LongMemEval (Wu et al., 2025): 500 questions; information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention; ~115K-token histories. BEAM: 1M–10M token scales. HaluMem (2026): memory consistency and hallucination. RealMem (2026): project-oriented interaction. MemGround (2026): gamified scenarios. Distinguish: all are answer-producing evaluations; MERIT is action-producing. Also position against long-context benchmarks (RULER, BABILong, InfiniteBench) which test single-pass attention, not write-and-retrieve loops.]

### 2.3 Tool-use and agent benchmarks
[Write 1 paragraph. τ-bench (Sierra) for tool-agent conversations with simulated users; AgentBench; SWE-bench for coding agents; WorkArena/WebArena for web tasks. Gap: these are (mostly) single-episode; cross-episode memory is not exercised. MERIT builds cross-episode dependency on top of a τ-bench-style environment.]

### 2.4 Reliability and failure analysis of agents
[Write 1 paragraph. Behavioral drift in long-horizon agents; production reliability as top challenge; confidently-wrong actions. Connect: stale-memory harm is a concrete, measurable instance of the reliability problem.]

## 3. The MERIT Benchmark

### 3.1 Design goals
G1 (Ecological validity): tasks require tool calls with verifiable end states, not free-text answers. G2 (Controllable dependency): the fraction of episodes requiring cross-episode memory is a benchmark parameter. G3 (Adversarial realism): memory corruption reflects real staleness processes (entity updates, rollbacks, contradictions). G4 (Cost transparency): every token in and out of the memory system is metered. G5 (Reproducibility): deterministic environment, seeded user simulator, pinned model versions, released traces.

### 3.2 Environment and task suite
We define an **episode** as one agent session with a simulated user and a tool API over a mutable world state (database). Episodes are grouped into **arcs** of E episodes sharing entities (a customer, a service, a household).

Domains:
- **D1 Customer Support (retail):** tools = order lookup, refund, address update, policy DB. Cross-episode facts: prior exceptions granted, updated addresses, promised follow-ups.
- **D2 IT Operations:** tools = log search, deployment history, config get/set, restart. Cross-episode facts: prior root causes, known-flaky services, config changes made last episode.
- **D3 Personal Assistant:** tools = calendar, email draft, preferences store. Cross-episode facts: standing preferences, previously scheduled commitments.

Each arc contains **dependent tasks** (success requires ≥1 fact from a prior episode; the fact is absent from the current episode's context and cannot be re-derived from tools alone or is expensive to re-derive) and **independent tasks** (solvable entirely within-episode). Ground-truth success is checked programmatically against the world state (e.g., refund issued to correct order at correct amount).

[Decide and state: number of arcs per domain (target ≥ 30), episodes per arc (target 4–6), dependent-task ratio grid {0%, 25%, 50%, 75%}. Total ≥ 500 scored episodes per condition.]

### 3.3 Memory conditions
- **C0 No memory:** each episode starts fresh (lower bound).
- **C1 Full replay:** all prior episode transcripts prepended (upper bound on information, upper bound on cost; truncate at model context limit, document truncation policy).
- **C2 Vector RAG:** chunked transcripts embedded; top-k retrieval per turn. [Pin: embedding model, chunk size, k.]
- **C3 Rolling summary:** per-episode LLM summary appended to a bounded memory document.
- **C4 Structured fact store:** LLM-extracted (entity, attribute, value, timestamp) facts with update-on-write semantics (Mem0-style, reimplemented for control).
- **C5 Hybrid:** C4 + C2 fallback.

All conditions share the same agent scaffold (ReAct-style tool loop), same prompts except the memory block, same tool APIs, and same decoding parameters (temperature 0 where supported; otherwise fixed seed sampling with S seeds).

### 3.4 Metrics
- **Task Success Rate (TSR):** programmatic end-state check, per condition, split by dependent vs. independent tasks.
- **Memory Utilization Rate (MUR):** among episodes where the required fact was present in retrieved memory, the fraction where the agent's action is consistent with that fact. Operationalized via (a) automatic value-tracing: the gold fact's value (order ID, amount, address) appears in the executed tool call arguments; (b) a stratified human audit of [n=100] episodes to validate the tracer (report agreement, e.g., Cohen's κ).
- **Ignore Rate:** 1 − MUR restricted to cases where memory was retrieved and correct.
- **Stale-Memory Harm (SMH):** TSR(clean memory) − TSR(corrupted memory) at corruption rate ρ ∈ {10%, 30%}; corruption types: stale value (superseded by a later true value), contradiction (two conflicting records), distractor (irrelevant but entity-matching). Also report **Verification Rate**: fraction of corrupted-memory episodes where the agent used a tool to re-verify before acting.
- **Cost:** input/output tokens and dollar cost per episode, decomposed into agent-loop cost vs. memory-system cost (write-time + read-time); wall-clock latency.
- **Cost-Adjusted Marginal Utility (CAMU):** ΔTSR relative to C0 divided by Δcost per episode relative to C0 (percentage points of success per dollar). Report the **break-even task value**: the dollar value a completed task must have for the memory condition to be economically rational.

### 3.5 Statistical methodology
Paired comparisons across conditions on identical task instances; 95% CIs via paired bootstrap (10,000 resamples) clustered at the arc level; S ≥ 3 seeds per condition; Holm–Bonferroni correction across the primary hypothesis family. Primary hypotheses (preregistered in the repo before running full experiments):
- H1: TSR(C2..C5) > TSR(C0) on dependent tasks.
- H2: TSR(C2..C5) ≤ TSR(C0) on independent tasks (distraction cost exists).
- H3: SMH > 0 for all conditions; structured memory (C4) has lower SMH than RAG (C2) due to update-on-write.
- H4: CAMU ordering differs from TSR ordering (the most accurate memory is not the most economical).

## 4. Experimental Setup
Models: [pin 3: one frontier API model, one mid-tier API model, one open-weight model ≥ 30B run via vLLM]. Decoding: [pin]. Harness: Python, LangGraph agent loop, LiteLLM proxy for unified metering; environment and tools implemented as deterministic Python services with seeded state. Compute: API credits ≈ $[fill from pilot]; open-weight model on 1×A100/H100 rented node. All prompts, seeds, and traces released at [repo URL].

## 5. Results  **[TEMPLATE — fill only with real measured numbers]**
5.1 Main effect of memory on dependent vs. independent tasks (Table 1, Fig. 1)
5.2 Utilization and Ignore Rates (Table 2) — include qualitative failure examples
5.3 Stale-Memory Harm and verification behavior (Fig. 2)
5.4 Cost decomposition and CAMU / break-even analysis (Fig. 3)
5.5 Model-scale interactions (does a stronger model need memory less, or exploit it better?)

## 6. Discussion  **[TEMPLATE]**
- Practitioner guidance: when to ship which memory architecture, as a function of dependency rate and task value.
- Why agents ignore correct memories (hypotheses grounded in observed traces).
- Implications for memory-system design: verification-before-action, provenance and timestamps.

## 7. Threats to Validity
Construct: simulated users and programmatic success checks may not capture all real-world success notions (mitigation: human audit sample). Internal: prompt differences across conditions minimized to the memory block only. External: three domains; results may not transfer to coding or web agents. Reproducibility: API model drift — pin model versions and dates; include open-weight model for permanent reproducibility.

## 8. Conclusion
[Write after results.]

## References  **[VERIFY every entry against the actual paper before submission]**
- Maharana et al., 2024. LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents.
- Wu et al., 2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.
- Chhikara et al., 2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. ECAI 2025.
- Packer et al., 2023. MemGPT: Towards LLMs as Operating Systems.
- Chen et al., 2026. HaluMem. [VERIFY]
- Bian et al., 2026. RealMem. [VERIFY]
- Tavakoli et al., 2026. BEAM. [VERIFY]
- Yao et al., 2023. ReAct: Synergizing Reasoning and Acting in Language Models.
- τ-bench: Yao et al., 2024. [VERIFY]
- Hsieh et al., 2024. RULER.
- Memory survey, 2026: "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers." arXiv:2603.07670. [VERIFY]
