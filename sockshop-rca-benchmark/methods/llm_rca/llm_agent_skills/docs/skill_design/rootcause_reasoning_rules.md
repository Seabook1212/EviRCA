# RootCauseReasoning Rules

## Overview

This document describes the coarse RCA heuristics used by `RootCauseReasoningSkill` in the V1 RCA framework.

The rules are practical troubleshooting priors. They help the framework reason about whether evidence looks local, propagated, shared, path-plausible, or likely to be operational noise. They are not deterministic truth, and they are not meant to replace evidence.

V1 remains a hybrid RCA system:

1. `MetricEvidenceSkill`, `LogEvidenceSkill`, and `TraceEvidenceSkill` extract structured evidence.
2. `RootCauseReasoningSkill` aggregates evidence at service and pod level.
3. Lightweight rules generate hints, score nudges, plausibility checks, and explanation notes.
4. LLM-assisted ranking, with heuristic fallback, produces the final ranked hypotheses.

The goal is to make common SRE, DevOps, and AIOps reasoning habits explicit and reviewable without turning the framework into a brittle expert system.

## Scope

These rules are intentionally coarse. They should be easy to explain in an incident review and easy to map into future implementation.

They should help answer questions such as:

- Is this a local pod problem or a shared service/dependency problem?
- Is this service the root cause or mostly a symptom carrier?
- Does the candidate fit the topology and timing of the incident?
- Is the evidence concrete enough to justify high confidence?
- Does the candidate explain the observed blast radius?

They should not:

- override strong direct evidence
- encode vendor-specific platform behavior
- require complex mathematics
- force every incident into one predefined pattern

## How The Rules Are Used

### Fault Matching

Rules can bias fault-type plausibility. For example:

- strong local CPU evidence supports `cpu_stress`
- memory pressure, OOM-like logs, or restart evidence supports `memory_stress` or `pod_failure`
- downstream edge latency and failure spikes support `network_delay`, `network_loss`, or dependency-related hypotheses
- concrete exception signatures support `exception_injection`

### Plausibility Checking

Rules can flag candidates that look less plausible:

- a candidate is outside the anomalous path
- a candidate appears much later than nearby impacted components
- a candidate has only symptom-level evidence
- a candidate is mostly supported by routine background logs

### Ranking Support

Rules can provide small ranking priors:

- boost strong local resource or failure evidence
- boost shared downstream dependency evidence
- reduce confidence for symptom-only or topology-conflicted candidates
- keep service-level and pod-level rankings aligned when pod evidence is more specific

### Explanation Support

Rules should produce short, defensible notes:

- "multiple sibling pods are similarly anomalous"
- "one pod is much stronger than its siblings"
- "trace anomalies are concentrated on downstream edges"
- "the candidate sits on or adjacent to the anomalous path"
- "evidence is mostly symptom-like and lacks strong local support"

## Rule Categories

Each category below contains practical rules and an operational interpretation. Some are already implemented as V1 hints in `checker.py` and `fault_matcher.py`; others are intended as future coarse hints.

### 1. Instance Distribution Rules

#### Motivation

The distribution of anomalies across sibling pods is often the fastest way to distinguish pod-local faults from shared service, dependency, or infrastructure problems.

#### Rules

1. If multiple pods of the same service become anomalous in a similar way at about the same time, prefer shared dependency, service-level, or shared downstream explanations over isolated pod-local faults.
2. If only one pod becomes strongly anomalous while sibling pods remain normal or much quieter, prefer pod-local fault explanations.
3. If only a small subset of pods is anomalous, consider traffic skew, hotspot routing, sticky sessions, uneven shard ownership, or node-local issues before concluding a true service-wide root cause.
4. Comparing an anomalous pod against healthy siblings is a strong local counterfactual and should be used whenever sibling data exists.

#### Practical Interpretation

- Broad sibling impact usually means something shared is wrong.
- One loud pod among quiet siblings usually means something local is wrong.
- Partial sibling impact is ambiguous and should reduce overconfidence.

#### Example

`carts-a` and `carts-b` both show similar latency spikes, while neither has strong local resource evidence. A shared downstream dependency is more plausible than two independent pod-local CPU faults.

`orders-a` shows high CPU, restart pressure, and local latency while `orders-b` is quiet. A pod-local fault should rank high.

### 2. Dependency Propagation Rules

#### Motivation

Microservice failures often propagate along dependency edges. The service that users notice first is not necessarily the service that caused the incident.

#### Rules

1. If multiple upstream services degrade and all depend on the same downstream service, database, cache, or queue, rank the shared downstream component higher.
2. If a service mainly shows outbound edge anomalies while local resource signals remain weak, treat it more as a propagated symptom carrier than a root-cause service.
3. If anomalies amplify along a call chain, earlier and deeper downstream anomalies are often more root-causal than upstream user-facing symptoms.
4. If several services show similar timeout or latency symptoms that point to the same dependency, prefer a shared-cause explanation over independent local failures.

#### Practical Interpretation

- Upstream services often accumulate visible symptoms.
- Shared dependencies can explain a broad blast radius with fewer assumptions.
- Outbound edge pain usually shifts suspicion toward the target or the path to it.

#### Example

`front-end`, `orders`, and `carts` all degrade. Traces from all three repeatedly point to `orders-db`. If `orders-db` also has local errors or resource pressure, it should outrank the upstream symptom carriers.

### 3. Temporal Reasoning Rules

#### Motivation

Timing is a strong RCA signal when it agrees with topology and evidence quality. Earlier evidence on a plausible propagation path often strengthens root-cause plausibility.

#### Rules

1. Earlier anomalous evidence on a plausible propagation path generally strengthens root-cause plausibility.
2. Much later anomalies usually suggest propagated symptoms, retries, saturation aftereffects, or secondary failures.
3. If many layers fail nearly simultaneously, consider shared infrastructure, rollout, configuration, traffic surge, or broad network causes.
4. Temporal ordering should not dominate evidence quality; early weak noise should not outrank later strong local failure evidence automatically.

#### Practical Interpretation

- Early plus specific plus path-plausible is strong.
- Late plus vague plus broad is weak.
- Simultaneous multi-layer degradation often points to something shared.

#### Example

`session-db` CPU and readiness anomalies appear before `user`, `orders`, and `front-end` latency. If those services depend on the affected path, `session-db` becomes a stronger root-cause candidate.

### 4. Topology And Path Rules

#### Motivation

A candidate should fit the anomalous dependency path. Topology is a guardrail against ranking unrelated noisy components too high.

#### Rules

1. Candidates outside the main anomalous path should be treated cautiously.
2. Candidates directly on or adjacent to anomalous edges or paths are more plausible than distant components.
3. Mere adjacency is not enough; adjacency must be supported by concrete metric, log, or trace evidence.
4. A good root-cause candidate should explain both the path of propagation and the observed blast radius.

#### Practical Interpretation

- Path fit matters, especially when the namespace is noisy.
- Topology support is a plausibility hint, not proof.
- Unrelated components need strong local evidence to stay high in the ranking.

#### Example

If abnormal traces repeatedly involve `shipping -> rabbitmq`, then `rabbitmq` is more plausible than a disconnected service with only mild latency noise.

### 5. Local Vs Propagated Signal Rules

#### Motivation

Not all evidence has the same diagnostic value. Strong local failure evidence is usually more root-causal than broad symptom evidence.

#### Rules

1. Strong local resource signals such as CPU, memory, restart count, readiness loss, OOM, crash, or kill evidence support local-fault hypotheses.
2. Latency or error symptoms without local supporting evidence should be treated cautiously as potential propagated symptoms.
3. Restart, readiness, OOM, crash, and exception evidence is usually stronger than generic latency spikes.
4. If a candidate mainly shows symptom KPIs and no local support, prefer dependency or shared-cause explanations unless other evidence is strong.
5. A local resource anomaly is strongest when it is materially elevated, temporally early, and localized to the candidate.

#### Practical Interpretation

- Local resource or crash evidence says "this component may be failing."
- Latency-only evidence often says "this component is hurt."
- Root-cause ranking should separate being impacted from being causal.

#### Example

`payment` shows latency spikes but no CPU, memory, restart, readiness, or exception evidence. `user-db` shows local CPU stress and database errors. `payment` should be treated more as impacted unless additional local evidence appears.

### 6. Log-Oriented Rules

#### Motivation

Logs vary widely in diagnostic value. Concrete failure signatures are useful; routine operational chatter can be misleading when counted as anomalies.

#### Rules

1. Explicit error signatures are more valuable than background listener, checkpoint, keepalive, or routine connection noise.
2. Bursts of repeated similar concrete errors are more meaningful than isolated generic errors.
3. Generic upstream timeout logs are less specific than concrete downstream failure logs.
4. Logs that name a failed dependency, rejected operation, exception type, crash event, OOM event, or connection failure deserve more weight than informational logs that merely accompany degraded behavior.
5. High-volume routine logs should be down-weighted unless they coincide with stronger local failure evidence.

#### Practical Interpretation

- "connection accepted" is weak.
- "connection refused by orders-db" is stronger.
- Repeated concrete failures beat generic distress messages.

#### Example

Repeated `timeout talking to orders-db` is useful, but repeated `request slow` messages are less specific. Routine database checkpoint logs should rarely dominate RCA on their own.

### 7. Trace-Oriented Rules

#### Motivation

Trace structure helps distinguish local execution problems from downstream, dependency, or network-related latency propagation.

#### Rules

1. Slow internal service spans suggest local service issues more than slow outbound edges do.
2. Slow outbound edges suggest downstream, dependency, or network-related issues more than local CPU faults do.
3. Failure spikes on key edges usually deserve more attention than mild latency-only anomalies.
4. Main critical request paths should weigh more than low-frequency side paths.
5. If one downstream edge repeatedly dominates abnormal trace latency, that dependency should receive stronger suspicion.

#### Practical Interpretation

- Slow internal spans point inward.
- Slow outbound edges point outward.
- Edge failures are usually more decisive than small latency shifts.
- Critical path traces are more representative of user impact than rare side paths.

#### Example

`orders` internal spans remain normal, but `orders -> orders-db` spans become slow and failure-prone. Suspicion should move toward `orders-db` or the path to it.

### 8. Service Role Rules

#### Motivation

Service role is not proof, but it is a useful prior. Databases, queues, caches, frontends, and stateless business services fail and surface symptoms differently.

#### Rules

1. Stateful shared dependencies such as databases, caches, queues, and message brokers deserve stronger suspicion when they show local anomalies.
2. Frontend, gateway, and API aggregation services often surface symptoms early, but they are not necessarily the root cause.
3. Stateless business services without local evidence are often victims or propagation points rather than origins.
4. Shared middleware with broad fan-in or fan-out can explain a wide blast radius and should be taken seriously when local evidence exists.
5. Role priors should bias investigation, not replace direct evidence.

#### Practical Interpretation

- A bad shared database can explain many bad services.
- A frontend with only timeout symptoms is often exposing downstream pain.
- Middleware deserves attention when many services rely on it.

#### Example

`rabbitmq` has local network and latency anomalies while several queue consumers degrade. The queue service is a stronger candidate than each consumer individually.

### 9. Comparative And Counterfactual Rules

#### Motivation

Good RCA reasoning is comparative. The question is not only "what looks bad?" but "what best explains the rest of the evidence?"

#### Rules

1. Comparing an anomalous pod against healthy siblings is a strong counterfactual for pod-local faults.
2. A good root-cause candidate should explain the observed blast radius, not just one local symptom.
3. When two candidates both fit, prefer the one with more specific, concrete, and local evidence.
4. When one candidate explains another candidate's symptoms through a plausible dependency path, the more explanatory candidate should usually rank higher.
5. If a candidate looks severe but cannot explain the wider impact, keep it as a local anomaly rather than over-promoting it.

#### Practical Interpretation

- Explanation power matters.
- Specific evidence beats vague evidence.
- A candidate that explains both itself and downstream symptoms is stronger than a candidate that only looks noisy.

#### Example

Both `orders` and `orders-db` look suspicious. `orders-db` has local resource evidence and explains `orders` symptoms through the dependency path, so `orders-db` should usually rank higher.

### 10. Blast Radius Rules

#### Motivation

The shape of impact is often as informative as the anomaly values themselves.

#### Rules

1. Broad synchronized impact should increase suspicion of shared dependencies, infrastructure, rollout, configuration, or network-wide causes.
2. Narrow isolated impact should increase suspicion of pod-local or service-local issues.
3. A candidate that naturally explains the breadth of impact should usually outrank one that explains only a narrow symptom.
4. If blast radius is broad but the candidate is narrow and weakly connected, treat it cautiously.
5. If blast radius is narrow but the candidate is a large shared dependency, require concrete local evidence before ranking it high.

#### Practical Interpretation

- Wide impact needs a wide explanation.
- Narrow impact often needs a local explanation.
- Blast radius is a sanity check against overfitting to one noisy signal.

#### Example

Five upstream services degrade together. One shared queue or database is often a more plausible explanation than five unrelated pod-local failures.

### 11. Node And Infrastructure Rules

#### Motivation

Not all incidents originate in application code or service dependencies. Some patterns are better explained by shared infrastructure.

#### Rules

1. If multiple unrelated services or pods on the same node become anomalous, consider node-level issues.
2. If impact is broad and highly synchronized, investigate shared infrastructure before only blaming business services.
3. If a small set of pods across different services all co-locate on the same node, consider noisy neighbor, node pressure, disk pressure, or node networking issues.
4. Infrastructure suspicion should increase when service-local evidence is weak but cross-service timing is tight.
5. Infrastructure hypotheses should still be supported by evidence such as node metrics, pod placement, network symptoms, or synchronized failures.

#### Practical Interpretation

- Co-location patterns matter.
- Synchronized cross-service degradation is often not accidental.
- Infrastructure candidates become stronger when application-specific evidence is weak.

#### Example

`orders-a`, `user-b`, and `catalogue-c` degrade together and share a node. A node-level issue becomes credible, especially if node CPU, memory, disk, or network evidence is also abnormal.

### 12. Confidence And Evidence Quality Rules

#### Motivation

Ranking confidence should reflect evidence quality, not only candidate order. A candidate can rank first while still having moderate confidence if evidence is weak or ambiguous.

#### Rules

1. Confidence should increase when evidence is local, concrete, repeated, temporally early, and topology-plausible.
2. Confidence should decrease when evidence is generic, late, symptom-only, topology-conflicted, or dominated by routine logs.
3. Multiple independent evidence sources strengthen confidence more than repeated copies of the same weak signal.
4. High confidence should require both good evidence and a plausible explanation of impact.
5. When evidence is ambiguous, ranking can still identify the best candidate, but the confidence should remain cautious.

#### Practical Interpretation

- Best-ranked does not always mean high-confidence.
- Concrete multi-source evidence should beat large volumes of weak evidence.
- Confidence should be calibrated enough to be defensible in review.

#### Example

A service with one generic timeout may still rank above weaker alternatives, but it should not receive the same confidence as a pod with CPU stress, restarts, and matching trace degradation.

## Short Practical Examples

### Example 1: Single Bad Pod

Evidence:

- `carts-1` shows strong CPU growth and local latency.
- `carts-2` remains healthy.
- traces do not show strong downstream concentration.

Interpretation:

- strong `single_pod_local_hint`
- strong local-fault prior
- `cpu_stress` should rank high

### Example 2: Shared Downstream Dependency

Evidence:

- `front-end`, `orders`, and `carts` all degrade.
- traces from each point to the same database edge.
- database logs show concrete failures.

Interpretation:

- strong `shared_issue_hint`
- strong `shared_downstream_dependency_hint`
- database should outrank upstream symptom carriers

### Example 3: Symptom-Only Frontend

Evidence:

- frontend latency rises.
- frontend has no local CPU, memory, restart, readiness, or exception evidence.
- downstream service has stronger local and trace evidence.

Interpretation:

- frontend is likely symptomatic
- downstream candidate should rank higher

### Example 4: Broad Synchronized Failure

Evidence:

- many services show anomalies almost simultaneously.
- local service evidence is weak or nonspecific.
- affected pods may span unrelated topology paths.

Interpretation:

- consider shared infrastructure, rollout, config, traffic, or network-level causes
- do not force a narrow single-service answer too early

### Example 5: Log Specificity

Evidence:

- one service emits many routine listener messages.
- another emits repeated concrete dependency failure messages.

Interpretation:

- routine chatter should be down-weighted
- concrete repeated failure signatures should receive more RCA weight

## Suggested Rule Names For Config Mapping

The following names are useful for future YAML and implementation mapping:

- `shared_multi_pod_anomaly`
- `single_pod_local_anomaly`
- `partial_pod_subset_anomaly`
- `traffic_skew_or_hotspot_hint`
- `downstream_edge_symptom`
- `shared_downstream_dependency`
- `call_chain_amplification_hint`
- `earlier_anomaly_precedence`
- `late_anomaly_penalty`
- `simultaneous_multi_layer_failure`
- `topology_conflict`
- `adjacent_path_support`
- `local_resource_support`
- `symptom_only_signal`
- `high_specificity_log_signal`
- `background_log_noise_penalty`
- `generic_timeout_penalty`
- `internal_span_local_issue`
- `outbound_edge_dependency_hint`
- `critical_path_weighting`
- `stateful_dependency_prior`
- `frontend_symptom_surface_prior`
- `blast_radius_fit_hint`
- `blast_radius_mismatch_penalty`
- `node_level_shared_failure`
- `synchronized_infrastructure_hint`
- `multi_source_evidence_support`

## Implementation Notes

These rules should remain coarse, explainable, and evidence-driven.

They are intended to generate:

- hints
- flags
- light score adjustments
- candidate notes
- ranking context for the LLM
- explanation text for reviewers

They should not become:

- a large deterministic expert system
- a vendor-specific operations playbook
- a replacement for metrics, logs, traces, and topology evidence
- a hard override of final ranking

For V1, the intended balance is:

1. Evidence extraction first.
2. Lightweight heuristic hints second.
3. LLM-assisted ranking or heuristic fallback last.

If a rule and direct evidence disagree, direct evidence should usually win. If several rules point in one direction but the evidence is weak, the system should treat the conclusion as plausible, not certain.

Future implementation should prefer small, inspectable flags such as `rule_hints`, `active_rules`, and explanation notes. This keeps `checker.py`, `fault_matcher.py`, and `ranker.py` debuggable while allowing the rule guide to grow gradually.
