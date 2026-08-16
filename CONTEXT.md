# EgoCUA-Bench Context

EgoCUA-Bench studies browser action during a user-initiated, task-scoped glasses session. The user is still manipulating objects, moving, or inspecting the scene while the Agent works on the web. The core problem is whether the Agent commits at the right time as egocentric evidence develops.

## Domain language

**Task-Scoped Glasses Session（任务级眼镜会话）**  
A user-initiated visual session with a clear start and end. It may provide a short sequence of egocentric observations while the task is active. It does not imply background or all-day recording.

**Progressive World Evidence（渐进现实证据）**  
Source-grounded visual or user evidence that becomes available in time order during the task. Later evidence must arise from the task timeline, not from a browser milestone.

**Candidate Commitment（候选承诺）**  
An action that changes state for a specific candidate, including selecting, saving, removing, routing to, or submitting it. Search and generic browsing are not commitments.

**Premature Commitment（过早承诺）**  
A candidate commitment made before the evidence required to distinguish the valid target has arrived.

**Productive Progress（安全推进）**  
Search, browsing, comparison, or other web work that remains valid before the final physical referent is known.

**Retain（保持）**  
Later evidence is irrelevant to the instructed referent, so the existing goal should remain unchanged.

**Revise（修正）**  
Later evidence invalidates the current referent or selection, so the Agent should undo the stale commitment and choose again.

**Integrate（整合）**  
New user information changes or extends the current goal during the same situated task.

**Clarify（澄清）**  
At the decision point, released evidence still supports multiple plausible referents and one minimal user answer can uniquely resolve them.

**Released Observation（发布观察）**  
The source-grounded part of the task-scoped session delivered to the Agent at a registered point on the task timeline. Keyframes are an input representation of a short visual interval, not separate user photographs.

**Glass-Use Necessity（眼镜场景必要性）**  
The admission requirement that the task depends on progressive, hands-free egocentric interaction and cannot be replaced by one task-start image plus a complete instruction without changing the measured decision.

**Source Event（来源事件）**  
An observed or annotation-supported object, action, order, or state change in continuous source media. Benchmark-authored speech must be declared separately and is not evidence of a naturally spoken source event.

**Target Granularity（目标粒度）**  
The finest identity level directly supported by released evidence: category, brand, variant, or exact instance.

**Diagnostic Ground Truth（分阶段真值）**  
The separate records for evidence sufficiency time, physical referent, acceptable environment entities, required final state, and actions that would count as premature or stale commitments.

**Experiment Run（实验运行）**  
One Agent attempt on one canonical task process and one replicate.

**Run Artifact（运行档案）**  
The prompts, released observations, user events, browser actions, questions, waits, final state, checker result, and errors from one run.

**Benchmark Release（基准发布版）**  
An immutable set of approved task timelines, observations, website snapshots, diagnostic ground truth, evaluator version, and provenance.

## Scope boundary

EgoCUA-Bench v1 measures response to a pre-registered task timeline. It does not measure autonomous event discovery, all-day memory, background surveillance, camera activation policy, or raw continuous-video streaming performance.

Graft, Refresh, and Clarify may remain as analysis labels for the primary decision in a task. They are project-defined labels, not a field-standard or exhaustive ontology.
