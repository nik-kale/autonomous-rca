# The Investigation Graph

The Investigation Graph is the primary output of the agentic RCA pipeline.
It is a directed acyclic graph (DAG) that captures every hypothesis, piece
of evidence, causal relationship, and reasoning step discovered during the
investigation.

## Why a Graph?

Traditional RCA tools produce a **linear report**: here's what happened,
here's the root cause, here's the fix.  This loses critical information:

- Which hypotheses were considered and rejected?
- What evidence supported or contradicted each hypothesis?
- How did the agents refine their thinking across iterations?
- What causal dependencies exist between failure modes?

The Investigation Graph retains all of this as **structured, queryable data**.
You can traverse it to understand not just *what* the root cause is, but
*how* the system arrived at that conclusion and *what else* it considered.

## Node Types

| Type | Description | Color (in visualization) |
|------|-------------|--------------------------|
| `problem` | The initial problem statement; DAG root | Red |
| `hypothesis` | A generated potential cause | Blue |
| `evidence` | A retrieved data point | Teal |
| `rejected` | A hypothesis disproven by evidence | Gray |
| `root_cause` | The converged conclusion; DAG leaf | Green |

Each node carries metadata:

```json
{
  "id": "h2",
  "type": "hypothesis",
  "text": "Pod eviction due to resource pressure on worker-3",
  "iteration": 1,
  "confidence": 0.85
}
```

## Edge Types

| Type | Meaning | Direction |
|------|---------|-----------|
| `generated_from` | Hypothesis derived from problem or evidence | problem → hypothesis |
| `supports` | Evidence confirms a hypothesis | evidence → hypothesis |
| `contradicts` | Evidence disproves a hypothesis | hypothesis → rejected |
| `causes` | Directed causal relationship | cause → effect |
| `depends_on` | B depends on A (A is deeper cause) | A → B |

## Dependency Pruning

The `depends_on` edge type enables **search space collapse**.  When the
Evaluation Agent discovers a causal chain:

```
DiskPressure (h4) ─── depends_on ──→ Pod Eviction (h2) ─── depends_on ──→ Service Unreachable (h1)
```

The deeper cause (disk pressure) subsumes the intermediate symptoms (pod
eviction, service unreachable).  The system can prune h1 and h2 as symptoms
and focus investigation on h4.

This is implemented through the `cause_of` field in the evaluation output:

```json
{
  "hypothesis_id": "h4",
  "confirmed": true,
  "cause_of": "h2",
  "confidence": 0.92,
  "reasoning": "DiskPressure=True on worker-3 triggered pod eviction"
}
```

## Accumulation via Reducers

The graph is not built in a single step.  It **accumulates** across
iterations via LangGraph's state reducer mechanism:

```python
nodes: Annotated[list[dict], operator.add]
edges: Annotated[list[dict], operator.add]
```

Each agent returns *new* graph fragments.  LangGraph appends them to the
running lists.  After N iterations, the `nodes` and `edges` lists contain
the complete Investigation Graph — every hypothesis ever considered, every
piece of evidence ever retrieved, and every relationship ever discovered.

### Iteration 1 (typical):
- 1 problem node + 3 hypothesis nodes + ~15 evidence nodes
- 3 `generated_from` edges + ~15 `supports` edges

### Iteration 2:
- 2 more hypothesis nodes + ~10 more evidence nodes + 1-2 rejected nodes
- `depends_on` edges emerge as causal relationships are detected

### Iteration 3 (convergence):
- 1 root_cause node + 1 synthesis node
- `causes` edges linking the root cause to the synthesis

## Querying the Graph

The graph is a list of dicts, so you can query it with simple list
comprehensions:

```python
# All confirmed hypotheses
confirmed = [n for n in nodes if n["type"] == "hypothesis"
             and any(e["to_id"] == n["id"] and e["type"] == "supports"
                     for e in edges)]

# The causal chain (dependency edges)
chain = [e for e in edges if e["type"] == "depends_on"]

# Evidence for a specific hypothesis
h2_evidence = [n for n in nodes if n["type"] == "evidence"
               and any(e["from_id"] == n["id"] and e["to_id"] == "h2"
                       for e in edges)]
```

For richer traversal, the visualization module builds a NetworkX `DiGraph`
from the raw lists:

```python
from src.graph.visualization import extract_root_cause_path

path = extract_root_cause_path(result["nodes"], result["edges"])
for step in path:
    print(f"[{step['type']}] {step['text']}")
```

## Export Formats

The graph can be exported as:

- **JSON** — `export_graph_json(nodes, edges)` returns a dict with nodes,
  edges, and summary statistics
- **PNG** — `plot_investigation_graph(nodes, edges)` renders a color-coded
  matplotlib figure
- **NetworkX DiGraph** — for programmatic analysis and traversal
