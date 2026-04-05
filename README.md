# Agentic Root-Cause Analysis with Investigation Graphs

A minimal but complete implementation of multi-agent autonomous root-cause
analysis using [LangGraph](https://langchain-ai.github.io/langgraph/),
demonstrating the **Investigation Graph** as a structured reasoning output.

## What This Is

This repository implements an architectural pattern for autonomous root-cause
analysis (RCA) in complex systems.  Four specialized agents collaborate in an
iterative loop — hypothesizing, gathering evidence, evaluating findings, and
synthesizing conclusions — while building a directed acyclic graph (DAG) that
captures every step of the reasoning process.

The Investigation Graph is not a side-effect of the investigation.  It **is**
the computation state: the graph grows with each iteration as the agents
accumulate nodes and edges through LangGraph's typed state reducers.

This is a pedagogical implementation — it uses mock evidence data and generic
LLM prompts.  It is not a production system or a framework.  It is a worked
example of a pattern you can adapt to your own infrastructure.

## Quick Start

```bash
git clone https://github.com/nik-kale/autonomous-rca.git
cd autonomous-rca
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
jupyter notebook notebooks/investigation_demo.ipynb
```

## The Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Diagnostic  │────▶│  Retrieval   │────▶│  Evaluation   │
│    Agent     │     │    Agent     │     │    Agent      │
└─────────────┘     └──────────────┘     └───────┬───────┘
       ▲                                          │
       │              ┌──────────┐                │
       └──────────────│ Converged?│◀──────────────┘
                      └─────┬────┘
                            │ Yes
                      ┌─────▼────┐
                      │ Analysis │
                      │   Agent  │
                      └─────┬────┘
                            │
                      ┌─────▼────┐
                      │  Result  │
                      └──────────┘
```

### The Four Agents

| Agent | Role | Source |
|-------|------|--------|
| **Diagnostic** | Generates 2-3 hypotheses targeting different domains (network, app, infra) | [`src/agents/diagnostic.py`](src/agents/diagnostic.py) |
| **Retrieval** | Gathers evidence using per-hypothesis strategy selection | [`src/agents/retrieval.py`](src/agents/retrieval.py) |
| **Evaluation** | Assesses evidence, detects causal dependencies between hypotheses | [`src/agents/evaluation.py`](src/agents/evaluation.py) |
| **Analysis** | Synthesizes the investigation into a root-cause report | [`src/agents/analysis.py`](src/agents/analysis.py) |

### The Investigation Graph

The output of every investigation is a DAG with typed nodes and edges:

**Node types:**
- `problem` — the initial problem statement (DAG root)
- `hypothesis` — a generated potential cause
- `evidence` — a retrieved data point
- `rejected` — a hypothesis disproven by evidence
- `root_cause` — the converged conclusion

**Edge types:**
- `generated_from` — hypothesis derived from problem or evidence
- `supports` — evidence confirms a hypothesis
- `contradicts` — evidence disproves a hypothesis
- `causes` — directed causal relationship
- `depends_on` — dependency relationship (enables pruning)

See [`docs/investigation_graph.md`](docs/investigation_graph.md) for a deep dive.

### Dependency Pruning

When the Evaluation Agent discovers that hypothesis A **caused** hypothesis B
(e.g., "disk pressure caused the pod eviction"), it emits a `depends_on` edge.
This collapses the search space: if A is the root cause, B is a symptom and
need not be investigated further.

The `cause_of` field in the evaluation output drives this:

```json
{
  "hypothesis_id": "h1",
  "confirmed": true,
  "cause_of": "h2",
  "confidence": 0.85,
  "reasoning": "DiskPressure on worker-3 caused checkout-pod-7b eviction"
}
```

### Retrieval Strategy Selection

Each hypothesis gets a retrieval strategy matched to its characteristics:

| Strategy | Best For | Implementation |
|----------|----------|---------------|
| Keyword | Known error codes, service names | [`src/retrieval/keyword.py`](src/retrieval/keyword.py) |
| TF-IDF | Rare patterns in large log corpora | [`src/retrieval/tfidf.py`](src/retrieval/tfidf.py) |
| Semantic | Fuzzy queries, synonym handling | [`src/retrieval/semantic.py`](src/retrieval/semantic.py) |

The [`strategy_selector`](src/retrieval/strategy_selector.py) chooses based
on hypothesis domain and evidence characteristics.

## Scenarios Included

1. **Kubernetes Disk Pressure** — Application outage caused by node storage
   exhaustion.  The investigation crosses application logs, Kubernetes events,
   and node metrics to trace from HTTP 503 → pod eviction → DiskPressure.

2. **Database Connection Pool** — Slow application caused by connection
   exhaustion from an unoptimized analytics query running without a timeout.

3. **TLS Certificate Expiry** — Intermittent 502 errors from a partial
   certificate rotation failure that left 2 of 5 backends with expired certs.

## Key Design Decisions

- **Typed state reducers** — `Annotated[list, operator.add]` for graph
  accumulation across iterations.  Each agent appends to the Investigation
  Graph rather than replacing it.

- **Conditional edges** — `should_continue` implements convergence detection:
  the loop exits when a root cause is confirmed with high confidence, all
  hypotheses are resolved, or the iteration budget is exhausted.

- **Pure node functions** — Each agent takes `InvestigationState` and returns
  a partial dict.  This makes agents independently testable and composable.

- **Graph = state** — The Investigation Graph IS the computation state, not a
  side-effect.  The same data structure drives the agents and serves as the
  output artifact.

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o-mini or better recommended)
- See [`requirements.txt`](requirements.txt) for full dependencies

## Running Tests

Tests use mock LLM responses and do not require an API key:

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
autonomous-rca/
├── src/
│   ├── state.py              # InvestigationState TypedDict
│   ├── agents/               # Four agent node functions
│   ├── graph/                # LangGraph builder + visualization
│   ├── retrieval/            # Multi-strategy retrieval
│   └── evidence/             # Synthetic mock data
├── scenarios/                # JSON scenario files
├── notebooks/                # Jupyter demo notebook
├── tests/                    # Pytest suite (no API key needed)
└── docs/                     # Architecture documentation
```

## Related

- Author: [Nik Kale](https://linkedin.com/in/nikkale)

## License

MIT — see [`LICENSE`](LICENSE).
