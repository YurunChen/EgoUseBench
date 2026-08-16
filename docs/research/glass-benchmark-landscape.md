# Glass / Egocentric Assistant Benchmark Landscape

_Last checked: 2026-08-09. This note only uses official papers, project pages, dataset pages, and official GitHub repositories._

## Bottom line

- [Ego2Web](https://ego2web.github.io/) is the closest public benchmark to EgoCUA for **open-web execution**.
- [EgoBench](https://arxiv.org/abs/2605.27820) is the most important additional execution neighbor surfaced in this search: it is the closest **interactive egocentric tool-using** benchmark here, but it evaluates tool use in a structured user-agent-tool environment rather than on live websites.
- The current wearable / smart-glasses landscape is split across different subproblems rather than one unified benchmark: latent goal inference ([WAGIBench](https://facebookresearch.github.io/WAGIBench/)), proactive interrupt / stay-silent decisions ([Wearable-AI / EgoProactive](https://huggingface.co/datasets/facebook/wearable-ai)), long-horizon streaming memory ([TeleEgo](https://teleai-uagi.github.io/TeleEgo/), [EGOSTREAM](https://saroo25.github.io/Egostream/)), deictic grounding / underspecified reference ([EgoPoint-Bench](https://guyyyug.github.io/EgoPoint-Bench/)), and static smart-glasses QA ([SUPERGLASSES](https://github.com/xandery-geek/SuperGlasses)).
- For EgoCUA's three task families, the strongest direct neighbors are:
  - `Graft`: WAGIBench, Wearable-AI / EgoProactive, EgoBench, TeleEgo
  - `Refresh`: Ego2Web, TeleEgo, EGOSTREAM, SUPERGLASSES, SAW-Bench
  - `Clarify`: EgoPoint-Bench, REI-Bench
- In the public work surfaced by this search, there is still no benchmark that jointly evaluates **first-person sensing + live web/tool execution + mid-trajectory goal revision + referent refresh + agent-initiated clarification**. That gap is exactly where EgoCUA can be novel.

## Local code archive

Official code snapshots have been placed under [`reference/`](../../reference/) for Ego2Web, WAGIBench, Wearable-AI Leaderboard, TeleEgo, EgoPoint-Bench, SUPERGLASSES, SAW-Bench, VisualWebArena, REI-Bench, EgoPlan-Bench, EgoSchema, and the adjacent Egocentric-Co-Pilot system. [`reference/README.md`](../../reference/README.md) records the exact commit, upstream URL, license status, and which large datasets were intentionally not downloaded.

EgoBench was not copied because no accessible official repository was found; its anonymous review URL currently requires authorization. WearVQA has an official dataset page but no separate evaluation-code repository surfaced in this search. EGOSTREAM lists its code as forthcoming. The 340 GB Wearable-AI dataset and starter kit remain access-gated, so only the public official leaderboard code is stored locally.

## Comparative map

| Item | Primary role | Real web/tool execution | Best match to EgoCUA | Main missing piece |
| --- | --- | --- | --- | --- |
| [Ego2Web](https://ego2web.github.io/) | Egocentric-to-web execution | Yes, live websites | `Refresh` + overall harness | No explicit user-driven goal updates or clarification turns |
| [EgoBench](https://arxiv.org/abs/2605.27820) | Interactive egocentric tool-using | Yes, structured tool env | Execution neighbor for `Graft` | Not live web; official code not publicly accessible in this search |
| [WAGIBench](https://facebookresearch.github.io/WAGIBench/) | Wearable latent-goal inference | No | `Graft` | No downstream execution |
| [Wearable-AI / EgoProactive](https://huggingface.co/datasets/facebook/wearable-ai) | Proactive interrupt-or-silent assistance | No | `Graft` + proactive timing | No browser / tool completion metric |
| [TeleEgo](https://teleai-uagi.github.io/TeleEgo/) | Streaming egocentric assistant memory | No | `Refresh` timing, long-horizon `Graft` | QA/memory only, not action execution |
| [EgoPoint-Bench](https://guyyyug.github.io/EgoPoint-Bench/) | Egocentric pointing / deixis grounding | No | `Clarify` | No dialogue or action loop |
| [SUPERGLASSES](https://github.com/xandery-geek/SuperGlasses) | Smart-glasses VQA | No | `Refresh` | Static image QA only |
| [WearVQA](https://huggingface.co/datasets/tonyliao-meta/WearVQA) | Wearable egocentric image QA | No | `Refresh` | Static image QA only; no separate official code repo surfaced |
| [SAW-Bench](https://sawbench.github.io/) | Continuous-video situated awareness | No | `Refresh` | Spatial QA only; no action loop |
| [EGOSTREAM](https://saroo25.github.io/Egostream/) | Streaming episodic memory with validity windows | No | `Refresh` protocol design | No public repo/data yet; not an action benchmark |
| [VisualWebArena](https://jykoh.com/vwa) | Multimodal web-agent execution | Yes, self-hosted realistic sites | Evaluation harness, trajectory judging | No egocentric sensing or dynamic intent |
| [REI-Bench](https://jcx0110.github.io/rei-bench-project) | Vague instruction grounding in planning | No, embodied simulator | `Clarify` | Household simulator, not first-person wearable |
| [EgoPlan-Bench](https://chenyi99.github.io/ego_plan/) | First-person next-action planning | No | Long-horizon planning | No live interaction |
| [EgoSchema](https://egoschema.github.io/) | Very long-form egocentric video understanding | No | Long-context `Refresh` memory pressure | No planning or execution |

## Layered view by evaluation mode

- `Inference / proactive decision`
  - [WAGIBench](https://facebookresearch.github.io/WAGIBench/): infer the latent user goal from multimodal wearable context.
  - [Wearable-AI / EgoProactive](https://huggingface.co/datasets/facebook/wearable-ai): decide at each chunk whether to stay silent or interrupt, and what to say.

- `QA / grounding`
  - [EgoPoint-Bench](https://guyyyug.github.io/EgoPoint-Bench/): deictic grounding under distractors.
  - [SUPERGLASSES](https://github.com/xandery-geek/SuperGlasses): smart-glasses visual QA with answer dynamism metadata.
  - [WearVQA](https://huggingface.co/datasets/tonyliao-meta/WearVQA): wearable-focused image QA under real-world egocentric quality degradation.
  - [SAW-Bench](https://sawbench.github.io/): observer-centric spatial QA over continuous egocentric video.
  - [EgoPlan-Bench](https://chenyi99.github.io/ego_plan/): first-person next-action planning as multiple-choice QA.
  - [EgoSchema](https://egoschema.github.io/): long-form egocentric temporal QA.

- `Streaming`
  - [TeleEgo](https://teleai-uagi.github.io/TeleEgo/): strict streaming memory with `RTA` / `MPT`.
  - [EGOSTREAM](https://saroo25.github.io/Egostream/): memory with explicit `Answer Validity Window`.

- `Tool-or-web execution`
  - [Ego2Web](https://ego2web.github.io/): grounded egocentric-to-web execution.
  - [EgoBench](https://arxiv.org/abs/2605.27820): egocentric interactive tool execution in a structured user-agent-tool environment.
  - [VisualWebArena](https://jykoh.com/vwa): realistic multimodal browser execution.
  - [REI-Bench](https://jcx0110.github.io/rei-bench-project): embodied execution in simulator, not web, but still action-level evaluation.

## Core wearable / glasses benchmarks

### 1. Ego2Web

Official links: [Project](https://ego2web.github.io/), [Paper](https://arxiv.org/abs/2603.22529), [Repo](https://github.com/Yui010206/Ego2Web), [Dataset](https://huggingface.co/datasets/Shoubin/Ego2Web)

- Research goal: benchmark web agents that must extract grounded evidence from egocentric video and use it to complete downstream online tasks.
- Experimental unit: one **video-instruction pair** executed by a web agent on live websites, with the trajectory judged against grounded real-world evidence.
- Input modalities: egocentric video, task instruction, webpage observations / screenshots / trajectory.
- Task generation: semi-automatic pipeline. A VLM parses first-person video into structured clip-level captions and visual metadata, an LLM generates grounded web tasks on active sites such as Amazon, YouTube, and Wikipedia, then humans verify and refine the example.
- Real web/tool execution: yes. This is the closest public benchmark to an actual glasses-to-web assistant loop.
- GT / judging: human-verified task plus annotated visual evidence; [Ego2WebJudge](https://github.com/Yui010206/Ego2Web) uses instruction, trajectory, screenshots, and grounded visual evidence. The project page reports about **84% agreement** with human judgment.
- Metrics: success rate plus judge-human agreement and ablations on video grounding.
- Dynamic / interactive ability: strong cross-modal grounding and online interaction; weak on explicit user updates. The benchmark assumes the instruction is fixed once issued.
- Scale: the paper reports **500 verified video-instruction pairs** across multiple live-site domains.
- Public code / data: code and dataset are public.
- License: the Hugging Face dataset card lists **MIT**; the GitHub repo had **no GitHub-recognized license file** as of 2026-08-09.
- EgoCUA relevance:
  - `Graft`: only partial. It proves a grounded-to-web execution loop, but not mid-run intent revision.
  - `Refresh`: strong. It already requires the agent to carry physical-world evidence into web execution.
  - `Clarify`: weak. No benchmarked clarification move.
- Not directly comparable:
  - No proactive question asking.
  - No explicit evolving referent after the task starts.
  - The web task is grounded in the source video, but the user is not a persistent live collaborator.

### 2. EgoBench

Official links: [Paper](https://arxiv.org/abs/2605.27820)

- Research goal: benchmark **interactive egocentric multimodal tool-using agents** that must jointly perform visual perception, tool-augmented multi-hop reasoning, and dynamic interaction with a user.
- Experimental unit: one egocentric-video-grounded task inside a **user-agent-tool interactive environment**.
- Input modalities: egocentric video, user dialogue, tool results, and environment / database state returned by the benchmark environment.
- Agent input / action:
  - the agent receives egocentric video-grounded user requests plus interaction history;
  - the agent emits natural-language dialogue and tool invocations;
  - the environment executes tools, updates structured database state, and returns the next observation.
- Task construction:
  - the paper describes a **three-stage synergistic pipeline**: egocentric video collection, scenario-specific tool / database construction, and task design;
  - each task is built to force multimodal reference grounding plus tool-based reasoning;
  - the task design explicitly combines multimodal perception complexity, logical reasoning complexity, and interaction complexity;
  - the paper states that every task includes at least one four-part structure: multimodal reference, tool retrieval for non-video attributes, conditional branching / filtering, and database modification through tools.
- Real web/tool execution: yes for tool execution, but **not** on open live websites. The environment is a structured tool-and-database setting.
- GT / judging:
  - the benchmark uses a simulated user with **Actor / Evaluator / Summarizer** roles;
  - it supports **Dynamic Easy**, **Dynamic Hard**, and **Static** interaction modes;
  - the evaluation is deterministic and checks both process-based and result-based equivalence.
- Metrics:
  - `ToolSucc`: whether all required tool calls are completed;
  - `MicroAcc`: fine-grained tool-call correctness over required operations;
  - `ResultSucc`: whether the final database state is correct;
  - `JointSucc`: whether the agent both uses tools correctly and reaches the correct final state.
- Dynamic / interactive ability: strong. This is the best execution-side neighbor in this search for interaction, tool use, and progressively revealed user information.
- Scale: **1,045 tasks** across **four daily scenarios**. The paper says **82.4%** of tasks use self-collected videos and **17.6%** use Ego4D videos.
- Public code / data: during this search, I found the official paper but **no accessible official public repo or dataset page**.
- License: not stated in the accessible official sources checked here.
- EgoCUA relevance:
  - `Graft`: strong. The dynamic interaction modes and simulated user are directly relevant to mid-task information release and revision pressure.
  - `Refresh`: moderate. The benchmark requires carrying grounded visual state into tool usage, but not explicit referent replacement over time.
  - `Clarify`: moderate. The interaction setup pressures active information elicitation, though not specifically first-person referent disambiguation.
- Not directly comparable:
  - It is a tool benchmark, not a live-web benchmark.
  - The simulated user is powerful and structured, which differs from a real human partner with minimal interventions.
  - Official code / dataset access was not publicly accessible in this search, so it is less reproducible than Ego2Web or SAW-Bench today.

### 3. WAGIBench

Official links: [Project](https://facebookresearch.github.io/WAGIBench/), [Paper / OpenReview](https://openreview.net/forum?id=REG4cJItSZ), [Repo](https://github.com/facebookresearch/WAGIBench)

- Research goal: measure how well assistive wearable agents can infer a user's latent goal from multimodal contextual observations, reducing the need for explicit user prompting.
- Experimental unit: one recording with ground-truth goal plus contextual modalities; models either answer a multiple-choice goal question or generate a free-form goal.
- Input modalities: egocentric video, audio, digital context, and longitudinal context.
- Task generation: newly collected benchmark with paired observations and ground-truth goals from real participants in wearable-assistant scenarios.
- Real web/tool execution: no. This is goal inference only.
- GT / judging: explicit reference goal. Official evaluation includes MCQ accuracy and a generative setting with LLM-judge alignment to human raters.
- Metrics: project page reports **93% human MCQ**, **84% best-model MCQ**, and only **55% relevant-goal rate** in the generative setting.
- Dynamic / interactive ability: strong latent-intent inference before action; no browser or tool loop, no clarification turn, no mid-trajectory goal revision.
- Scale: **29 hours**, **348 participants**, **3,477 recordings**.
- Public code / data: official evaluation repo and dataset download links are public.
- License: repo is **Apache-2.0**. The project page / README do not clearly surface a separate dataset license.
- EgoCUA relevance:
  - `Graft`: strong conceptual neighbor. It operationalizes "the user's real goal is not fully verbalized yet".
  - `Refresh`: moderate, because extra modalities can disambiguate latent intent over time.
  - `Clarify`: weak, since the benchmark measures inference instead of asking.
- Not directly comparable:
  - No downstream web or tool execution.
  - The main uncertainty is latent goal discovery, not revising a previously committed action sequence.

### 4. Wearable-AI Challenge / EgoProactive

Official links: [Dataset](https://huggingface.co/datasets/facebook/wearable-ai), [Leaderboard](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard), [Paper](https://arxiv.org/abs/2606.04970)

- Research goal: benchmark **proactive procedural assistance** from wearable-egocentric video, especially the policy of **when to interrupt, when to stay silent, and how to recover from deviations**.
- Experimental unit:
  - Challenge-level tracks: `EgoConv`, `EgoLongQA`, `EgoProactive`.
  - Most relevant to EgoCUA: one **EgoProactive streaming session** broken into sequential 8-second chunks.
- Input modalities: egocentric video stream plus available past context only; the official leaderboard rules enforce **no-future-leak**.
- Agent input / action:
  - For `EgoProactive`, the agent must output either the literal string `$silent$` or an utterance prefixed by `$interrupt$...` for each 8-second chunk.
  - For `EgoConv`, the agent produces multi-turn free-form answers.
  - For `EgoLongQA`, the agent predicts `A/B/C/D`.
- Task construction:
  - The PWR paper introduces **EgoProactive**, a wearable-first procedural dataset with explicit Out-of-Plan (`OOP`) annotations and recovery utterances.
  - The paper reports **700 smart-glasses videos** and **9,935 evaluation instances**.
  - The accessible paper text is internally inconsistent about the exact OOP count: the introduction mentions **1,833** OOP deviation + recovery instances, while the main dataset section reports **3,433** OOP scenarios. EgoCUA should not rely on that exact OOP count without a clarified benchmark release.
  - The challenge package on Hugging Face exposes the benchmark as the gated dataset `facebook/wearable-ai`.
- Real web/tool execution: no. This is proactive assistive decision-making over video, not browser or tool completion.
- Evaluator:
  - The official leaderboard code scores `EgoProactive` with **Macro F1** over the binary decision `interrupt` vs `silent`.
  - The paper also reports richer proactive metrics such as OOP detection recall and recovery quality, but those are not the public leaderboard's headline metric.
- Dynamic / interactive ability: strongest public benchmark here for proactive interruption policy and recovery timing; still not a downstream digital-action benchmark.
- Scale: Hugging Face dataset page lists **340 GB**, **auto-gated** access, and a **CC BY-NC 4.0** license.
- Public code / data:
  - Official public leaderboard code is accessible through the Hugging Face space.
  - Dataset access is gated.
  - The starter kit is not openly pullable without access to the gated dataset package.
- License: dataset page lists **CC BY-NC 4.0**; the public leaderboard space is **Apache-2.0**.
- EgoCUA relevance:
  - `Graft`: strong. It is the closest public benchmark here to "should the assistant proactively revise / inject guidance right now?"
  - `Refresh`: moderate. The interrupt decision is tied to current procedural state, not long-term factual replacement.
  - `Clarify`: weak directly, but it establishes a benchmarked policy for deciding whether speaking at all is worthwhile.
- Not directly comparable:
  - No real web or external tool completion.
  - The supervision is proactive coaching on procedures, not grounded disambiguation of a user's referent on the fly.

### 5. TeleEgo

Official links: [Project](https://teleai-uagi.github.io/TeleEgo/), [Paper](https://arxiv.org/abs/2510.23981), [Repo](https://github.com/TeleAI-UAGI/TeleEgo), [Dataset](https://huggingface.co/datasets/David0219/TeleEgo)

- Research goal: benchmark egocentric AI assistants in the wild under **strict streaming** long-term memory and reasoning conditions.
- Experimental unit: one streaming question over a multi-day egocentric record, with time-sensitive answering and optional delayed recall tests.
- Input modalities: egocentric video, speech / conversations, narration, event descriptions.
- Task generation: five participants are recorded for three days each across four daily scenarios; expert annotators create QA over 12 task types.
- Real web/tool execution: no. This is a streaming memory-and-reasoning benchmark, not an action benchmark.
- GT / judging: manually annotated answers. The released framework includes strict streaming metrics and recall probes.
- Metrics:
  - `RTA` (Real-Time Accuracy): percentage answered correctly within **5 seconds**.
  - `MPT` (Memory Persistence Time): how long a model keeps a fact after first using it.
  - The official docs explicitly say **MPT is not yet reliably tabled** because current model/session constraints make the measurement unstable.
- Dynamic / interactive ability: very strong on online timing and memory persistence; still not an action-selection benchmark.
- Scale: official docs / README describe **70+ hours**, **5 participants**, **3,291 manually annotated questions**, **12 cognitive tasks**.
- Public code / data: repo, leaderboard, and dataset are public.
- License: README states **MIT** for code and **research-only** for dataset usage.
- EgoCUA relevance:
  - `Graft`: moderate. It evaluates delayed intent inference and memory-based goal use, not explicit instruction rewriting.
  - `Refresh`: strong. The strict streaming protocol, `RTA`, and `MPT` are directly relevant to "when should old evidence be retained vs replaced".
  - `Clarify`: weak. The benchmark evaluates answering under memory pressure, not clarification behavior.
- Not directly comparable:
  - No real website or tool actions.
  - No notion of browser-side task completion.
  - The agent is tested as a memory-heavy QA assistant, not a planner that changes commitments mid-execution.

### 6. EgoPoint-Bench

Official links: [Project](https://guyyyug.github.io/EgoPoint-Bench/), [Paper](https://arxiv.org/abs/2604.21461), [Repo](https://github.com/GUYYYUG/EgoPoint), [Dataset](https://huggingface.co/datasets/GUYYYUG/EgoPoint)

- Research goal: test whether multimodal models can correctly ground **pointing-based deictic reference** in egocentric vision instead of hallucinating nearby or salient objects.
- Experimental unit: one egocentric pointing QA sample.
- Input modalities: egocentric image, pointing gesture, natural-language question. The benchmark spans explicit to implicit deictic expressions.
- Task generation: dual-source pipeline with scalable simulation (`Point-Sim`) plus real-world smart-glasses-style capture and capability-oriented annotation.
- Real web/tool execution: no.
- GT / judging: objective QA targets across multiple-choice, true/false, and open-ended questions.
- Metrics: task accuracy by capability slice and by real vs simulated split; the benchmark is built to expose "referential hallucination".
- Dynamic / interactive ability: very strong on referent grounding under distractors and deixis ambiguity; no dialogue turn and no downstream action execution.
- Scale: **11,729 samples** total, split into **10,567 simulation** and **1,162 real-world** samples; organized by **5 capability dimensions** and **3 deixis levels**.
- Public code / data: official repo and dataset are public.
- License: **Apache-2.0**.
- EgoCUA relevance:
  - `Graft`: weak.
  - `Refresh`: moderate, because it stresses whether the currently attended referent is correct under nearby distractors.
  - `Clarify`: strong. This is the cleanest official egocentric benchmark here for underspecified reference and distractor-heavy grounding.
- Not directly comparable:
  - The action space ends at answering, not browsing or tool use.
  - It does not test whether the agent asks a clarifying question when confidence is low.

### 7. SUPERGLASSES

Official links: [Paper](https://arxiv.org/abs/2602.22683), [Repo](https://github.com/xandery-geek/SuperGlasses), [Dataset](https://huggingface.co/datasets/xandery/SuperGlasses)

- Research goal: evaluate VLMs as intelligent agents for AI smart glasses using real-world data captured entirely by smart-glasses devices.
- Experimental unit: one egocentric image-question pair.
- Input modalities: image plus question; metadata also includes language, hops, location, image quality, tool annotations, and a `dynamism` label.
- Task generation: real-world smart-glasses capture across 14 domains and 8 query categories.
- Real web/tool execution: no direct execution. The dataset contains retrieval-tool annotations in metadata, but does not run tools.
- GT / judging: answer string plus metadata slices.
- Metrics: **F1-Recall** and **LLM Judge Accuracy**.
- Dynamic / interactive ability: static one-shot QA, but the explicit `dynamism` field is useful because it tags whether answers are temporally stable or fast-changing.
- Scale: **2,422 image-question pairs**.
- Public code / data: code and dataset are public.
- License: dataset **CC BY 4.0**, code **MIT**.
- EgoCUA relevance:
  - `Graft`: weak.
  - `Refresh`: moderate-to-strong. The dataset explicitly labels answer dynamism, which is useful for deciding when visual evidence should be refreshed.
  - `Clarify`: weak-to-moderate. It contains referential questions but not an interactive clarification protocol.
- Not directly comparable:
  - No video stream.
  - No browser or tool execution.
  - No persistent user goal over time.

### 8. WearVQA

Official links: [Dataset](https://huggingface.co/datasets/tonyliao-meta/WearVQA), [Paper](https://openreview.net/forum?id=s5p9ByKN1j)

- Research goal: benchmark VQA for wearable assistants in **egocentric authentic real-world scenarios**, especially where wearable images are degraded by occlusion, blur, poor lighting, and similar capture artifacts.
- Experimental unit: one image-question-answer triplet.
- Input modalities: egocentric image plus natural-language question.
- Task construction:
  - the official dataset card describes **2,520** image-question-answer triplets;
  - the examples span **7 image domains**, **10 cognitive task types**, and **6 wearable-specific image quality issues**;
  - questions are designed to be answerable from the image plus common sense.
- Real web/tool execution: no.
- GT / judging: ground-truth answer strings; the official dataset card says the benchmark is paired with an **LLM-as-a-judge** framework with **96% labeling accuracy**.
- Metrics: answer correctness plus judge-based evaluation slices; the official dataset card also mentions a leaderboard, though the accessible dataset README does not expose a standalone evaluation repo.
- Dynamic / interactive ability: static QA only. Its value is realism of wearable-image degradation rather than interaction.
- Scale: **2,520** triplets.
- Public code / data: the official public artifact surfaced in this search is the Hugging Face dataset page; I did **not** find a separate official public code repository.
- License: the official Hugging Face dataset card lists **Apache-2.0**.
- EgoCUA relevance:
  - `Graft`: weak.
  - `Refresh`: moderate. It is a useful source of wearable-specific perceptual failures that can make stale or low-quality observations unreliable.
  - `Clarify`: weak. It does not benchmark interaction or referent questioning.
- Not directly comparable:
  - Static image QA only.
  - No action loop, no dialogue, no web / tool execution.

## Adjacent benchmarks that fill EgoCUA-specific gaps

### 9. SAW-Bench

Official links: [Project](https://sawbench.github.io/), [Paper](http://arxiv.org/abs/2602.16682), [Repo](https://github.com/UCSB-AI/SAW-Bench), [Dataset](https://huggingface.co/datasets/ucsbai/SAW-Bench)

- Research goal: evaluate **observer-centric situated awareness** in multimodal models from continuous egocentric video, rather than only environment-centric object relations.
- Experimental unit: one egocentric video plus one multiple-choice QA item.
- Input modalities: egocentric video only. The project page and repo explicitly say audio is removed, and models do not receive any bird's-eye / global scene representation.
- Task construction:
  - **786 real-world videos** captured with **Ray-Ban Meta Gen 2** smart glasses;
  - **2,071 human-annotated QA pairs** over **six tasks**: self-localization, relative direction, route shape, reverse route plan, spatial memory, spatial affordance;
  - most clips are continuous single recordings; the spatial-memory task concatenates before/after clips of the same scene after a controlled change.
- Real web/tool execution: no.
- GT / judging: multiple-choice answers parsed into answer letters and scored by accuracy.
- Metrics: per-task and overall QA accuracy. The project page reports that even the best model trails humans by **37.66%**.
- Dynamic / interactive ability: strong on continuous observer-centric state tracking; no tool or browser actions.
- Scale: **786 videos**, **2,071 QA pairs**.
- Public code / data: official code and dataset are public. The repo includes download scripts and a full `generate → parse → score` evaluation pipeline.
- License: the repo README states **CC BY-NC 4.0** for code and data, even though GitHub API did not surface a standard SPDX license record.
- EgoCUA relevance:
  - `Graft`: weak.
  - `Refresh`: strong conceptual neighbor for observer-centric state refresh, path memory, and scene change detection.
  - `Clarify`: weak.
- Not directly comparable:
  - This is still QA over video, not agent execution.
  - It focuses on spatial state, not user-intent revision or interactive clarification.

### 10. EGOSTREAM

Official links: [Project](https://saroo25.github.io/Egostream/), [Paper](https://arxiv.org/abs/2605.31557)

- Research goal: diagnostic benchmark for **streaming episodic memory** in egocentric vision, with explicit modeling of when an answer should stop being considered valid as the world changes.
- Experimental unit: one curated memory question expanded into many recall-conditioned evaluations.
- Input modalities: egocentric video stream plus question at a specified recall time.
- Task generation: curated questions are expanded using the **Answer Validity Window (AVW)**, which defines the time span during which a given answer remains valid.
- Real web/tool execution: no.
- GT / judging: answer correctness conditioned on recall time and AVW.
- Metrics: accuracy across recall regimes and memory categories; the paper reports aggregate ceilings around **45%** on tested methods and emphasizes sub-real-time throughput.
- Dynamic / interactive ability: strong on modeling world-state change over time; no browser or tool execution.
- Scale: **2,250 curated questions** expanded into **8,528 recall-conditioned evaluations** across seven cognitive dimensions.
- Public code / data: as of the public project page checked on 2026-08-09, **data is not publicly released yet** and **GitHub is listed as coming soon**.
- License: not publicly stated on the project page.
- EgoCUA relevance:
  - `Graft`: weak.
  - `Refresh`: very strong. `AVW` is a directly useful design primitive for Refresh-style retain/replace ground truth.
  - `Clarify`: weak.
- Not directly comparable:
  - Memory benchmark only.
  - No action execution and no user-initiated repair loop.

### 11. VisualWebArena

Official links: [Project](https://jykoh.com/vwa), [Paper](https://arxiv.org/abs/2401.13649), [Repo](https://github.com/web-arena-x/visualwebarena)

- Research goal: evaluate multimodal autonomous agents on realistic visual web tasks with execution-based scoring.
- Experimental unit: one web task executed on self-hosted realistic sites.
- Input modalities: webpage observations, screenshots, accessibility tree / DOM views, text instruction, and sometimes reference visual content.
- Task generation: task templates over realistic sites such as classifieds, shopping, Reddit, Wikipedia, and homepage environments; built on top of the WebArena execution framework.
- Real web/tool execution: yes.
- GT / judging: execution-based evaluation via environment end states and trajectories; human trajectories are provided for a 233-task subset.
- Metrics: success rate; the repo notes about **89% human success** on the 233-task human-eval subset.
- Dynamic / interactive ability: very strong on realistic browser execution; no egocentric physical-world input and no explicit changing user goal.
- Scale: **910 tasks** in the released benchmark; **233 template types** referenced by the human-trajectory subset.
- Public code / data: repo and setup scripts are public; released trajectories are also public.
- License: **MIT**.
- EgoCUA relevance:
  - `Graft`: weak directly, but excellent for harness design.
  - `Refresh`: moderate, mainly because visual inputs can matter for shopping / matching style tasks.
  - `Clarify`: weak.
- Not directly comparable:
  - No wearable perspective.
  - No changing user belief / referent state outside the browser.

### 12. REI-Bench

Official links: [Project](https://jcx0110.github.io/rei-bench-project), [ICLR 2026 paper](https://openreview.net/forum?id=vmBIF25KLf), [Repo](https://github.com/jcx0110/REI-Bench), [arXiv](https://arxiv.org/abs/2505.10872)

- Research goal: evaluate whether embodied agents can understand **vague, underspecified, or coreferential instructions** in task planning.
- Experimental unit: one ALFRED / AI2-THOR planning example under a controlled vagueness condition.
- Input modalities: natural-language instruction or multi-turn context grounded in an embodied household simulator.
- Task generation: REI-Bench systematically transforms planning tasks to include pragmatic vagueness in referring expressions.
- Real web/tool execution: no live web; execution is embodied-simulator planning.
- GT / judging: task success in planner execution plus analysis by vagueness condition.
- Metrics: success rate; the latest official project and ICLR paper report up to **36.9%** drops from vagueness. An older arXiv metadata view still surfaced a higher drop number, so the latest official project / ICLR figure is the safer one to quote.
- Dynamic / interactive ability: strongest direct benchmark in this list for `Clarify`-like underspecification, but it mainly studies better disambiguation and prompting rather than an agent explicitly asking the human a question.
- Scale: official repo and project emphasize the ALFRED-derived evaluation split and condition design, but do **not** foreground a single benchmark-wide sample count as clearly as the paper's qualitative summaries do.
- Public code / data: official evaluation code is public; evaluation depends on ALFRED / AI2-THOR assets.
- License: the repo had **no GitHub-recognized license** as of 2026-08-09.
- EgoCUA relevance:
  - `Graft`: weak.
  - `Refresh`: weak.
  - `Clarify`: strong conceptual neighbor.
- Not directly comparable:
  - Household simulator instead of egocentric wearable footage.
  - No browser or external tools.
  - Clarification is mostly implicit in planner design, not benchmarked as a conversational move.

### 13. EgoPlan-Bench

Official links: [Project](https://chenyi99.github.io/ego_plan/), [Paper](https://arxiv.org/abs/2312.06722), [Repo](https://github.com/ChenYi99/EgoPlan), [Data](https://drive.google.com/drive/folders/1qVtPzhHmCgdQ5JlMeAL3OZtvbHaXktTo?usp=sharing)

- Research goal: benchmark human-level planning in egocentric video, where the model must infer the next feasible action plan from task progress.
- Experimental unit: one multiple-choice next-action planning question.
- Input modalities: progress video, current visual observation, open-form task goal, and candidate action plans.
- Task generation: derived from Epic-Kitchens and Ego4D; GPT-4 identifies hierarchical task goals and generates candidate actions, then 10 expert annotators verify each multiple-choice question.
- Real web/tool execution: no.
- GT / judging: verified correct next action among multiple-choice candidates.
- Metrics: multiple-choice accuracy.
- Dynamic / interactive ability: strong for long-horizon first-person planning state; no user interaction, no clarification, no external tool execution.
- Scale: **4,939 multiple-choice questions**, **234 root verbs**, **558 unique objects**, **419 scenes**.
- Public code / data: code and leaderboard are public; dataset release points to public data files.
- License: repo is **BSD-3-Clause**. The paper states the benchmark is built on Epic-Kitchens (**CC BY-NC 4.0**) and Ego4D (**Ego4D license**).
- EgoCUA relevance:
  - `Graft`: moderate as a planning substrate, not as a changing-goal benchmark.
  - `Refresh`: moderate because the model must update plan state from current observation.
  - `Clarify`: weak.
- Not directly comparable:
  - No persistent dialogue or dynamic user revision.
  - No browser / tool environment.

### 14. EgoSchema

Official links: [Project](https://egoschema.github.io/), [Paper](https://arxiv.org/abs/2308.09126), [Repo](https://github.com/egoschema/EgoSchema), [Kaggle benchmark](https://www.kaggle.com/competitions/egoschema-public/overview)

- Research goal: diagnostic benchmark for **very long-form** video-language understanding.
- Experimental unit: one three-minute video clip plus one five-way multiple-choice question.
- Input modalities: egocentric video plus question.
- Task generation: human-curated from Ego4D, with emphasis on high intrinsic temporal reasoning length.
- Real web/tool execution: no.
- GT / judging: multiple-choice answer; only a **500-question** answer subset is public offline, while full-benchmark evaluation goes through Kaggle / validation server.
- Metrics: MCQ accuracy.
- Dynamic / interactive ability: strong long-context perception pressure; no planning or action loop.
- Scale: **5,031 questions** over **250+ hours** of egocentric video.
- Public code / data: repo and evaluation instructions are public; videos are downloadable through Kaggle / Wasabi / Google Drive workflows.
- License: the official arXiv abstract states data and zero-shot evaluation code are open-sourced under the **Ego4D license**; the GitHub repo itself had **no GitHub-recognized separate license** as of 2026-08-09.
- EgoCUA relevance:
  - `Graft`: weak.
  - `Refresh`: strong as a pure long-context perceptual stress test.
  - `Clarify`: weak.
- Not directly comparable:
  - No online action.
  - No user-agent interaction policy.

## System reference, not benchmark

### Egocentric-Co-Pilot

Official links: [Paper](https://arxiv.org/abs/2603.01104), [Repo](https://github.com/YoungSeng/Egocentric-Co-Pilot)

- Why include it: this is not a benchmark, but it is a directly relevant **system reference** for a web-native smart-glasses agent with tool execution, real-time audio, and a proactive clarifier module.
- Confirmed capabilities from the official repo:
  - tool execution / MCP-style integration
  - Android front-end with real-time visual perception and WebSocket bridge
  - explicit **plug-and-play clarifier** support
- License: the repo had **no GitHub-recognized license** as of 2026-08-09, so it should be treated as **read-only reference**, not copied into distributed benchmark code without further permission review.
- EgoCUA relevance:
  - good inspiration for runtime architecture and proactive clarification wiring
  - not valid evidence for benchmark design, because it does not define a public evaluation protocol

## What this implies for EgoCUA

### 1. The clearest novelty claim

In the public work surfaced by this search, EgoCUA can honestly claim to combine five ingredients that are currently separated across different official benchmarks:

- egocentric / wearable perception (`Ego2Web`, `TeleEgo`, `SUPERGLASSES`, `WearVQA`, `EgoPoint-Bench`, `SAW-Bench`)
- proactive speak-or-stay-silent policy (`Wearable-AI / EgoProactive`)
- interactive tool or web execution (`Ego2Web`, `EgoBench`, `VisualWebArena`)
- dynamic world-state validity over time (`TeleEgo`, `EGOSTREAM`)
- underspecified referent handling (`EgoPoint-Bench`, `REI-Bench`)

### 2. Concrete design takeaways for `Graft / Refresh / Clarify`

- `Graft`
  - Borrow WAGIBench's framing that the user's true goal may be latent or only partially verbalized.
  - Borrow Wearable-AI / EgoProactive's explicit `interrupt` vs `silent` supervision so proactive intervention itself is benchmarked, not just final answer quality.
  - Borrow EgoBench's `Dynamic Easy / Dynamic Hard / Static` interaction settings so the same task can be stressed under progressively more hostile information-release conditions.
  - Borrow TeleEgo's strict streaming timing so revised goals are judged relative to when evidence becomes available.
  - Do **not** reduce Graft to "predict the hidden goal before acting"; EgoCUA should require revising an already-started commitment.

- `Refresh`
  - Borrow EGOSTREAM's `Answer Validity Window` idea almost directly, but translate it from QA answers to referent / action validity windows.
  - Borrow SUPERGLASSES' `dynamism` notion so each task explicitly marks whether stale visual evidence should be retained or overwritten.
  - Borrow SAW-Bench's observer-centric state formulation so Refresh is not just "new frame arrived", but "the agent's internal spatial / referential state should or should not change".
  - Use Ego2Web-style web execution so Refresh is not only about memory, but about whether the agent changes the downstream digital action.

- `Clarify`
  - Borrow EgoPoint-Bench's deictic levels and distractor-heavy pointing setup for first-person grounding.
  - Borrow REI-Bench's pragmatic view that ambiguity is not just missing nouns; it often lives in referring expressions and dialogue history.
  - Add an explicit **permitted clarification act** and score whether one concise question uniquely resolves the target, which the current public benchmarks do not operationalize well.

### 3. What should not be claimed by analogy

- WAGIBench is not evidence that current systems can revise goals mid-execution; it is pre-action goal inference.
- Wearable-AI / EgoProactive is not a web benchmark; it justifies proactive interruption policy and recovery timing, not browser completion.
- EgoBench is not evidence for live-web robustness; it justifies interactive tool-using evaluation under egocentric grounding in a structured environment.
- TeleEgo and EGOSTREAM are not action benchmarks; they justify streaming-memory protocols, not browser success metrics.
- SAW-Bench and WearVQA are not agent benchmarks; they justify perception and observer-centric state pressure, not tool execution.
- EgoPoint-Bench and REI-Bench are not proactive-clarification benchmarks in the full assistant sense; they measure ambiguity handling, mostly without a live human repair loop.
- EgoSchema and EgoPlan-Bench justify long-context understanding and planning pressure, but not web interaction.

## Recommended official repos to clone

### Highest priority

- [Yui010206/Ego2Web](https://github.com/Yui010206/Ego2Web)
  - Closest public benchmark to EgoCUA's end-to-end problem.
  - Useful for task schema, judge design, and egocentric-to-web execution framing.

- [UCSB-AI/SAW-Bench](https://github.com/UCSB-AI/SAW-Bench)
  - Best public continuous egocentric-video benchmark here for observer-centric situated awareness.
  - Useful for Refresh-style spatial state updates, continuous-video packaging, and reproducible evaluation code.

- [web-arena-x/visualwebarena](https://github.com/web-arena-x/visualwebarena)
  - Best public execution harness for realistic multimodal browser tasks.
  - Useful for evaluator plumbing, site setup, and trajectory handling.

- [facebookresearch/WAGIBench](https://github.com/facebookresearch/WAGIBench)
  - Best public wearable-goal-inference benchmark.
  - Useful for multimodal context packaging and latent-intent baselines.

- [TeleAI-UAGI/TeleEgo](https://github.com/TeleAI-UAGI/TeleEgo)
  - Best public streaming wearable-memory benchmark.
  - Useful for `RTA` / `MPT` timing protocol ideas and long-horizon egocentric QA format.

- [GUYYYUG/EgoPoint](https://github.com/GUYYYUG/EgoPoint)
  - Strongest public egocentric deictic-grounding benchmark.
  - Useful for distractor construction, deixis levels, and Clarify-style referent ambiguity design.

- [facebook/wearable-ai-leaderboard](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard)
  - Public official code for the Wearable-AI challenge leaderboard and submission contract.
  - Useful for exact `EgoConv` / `EgoLongQA` / `EgoProactive` output formats, `Macro F1` scoring expectations, and the no-future-leak evaluation contract.

### Medium priority

- [jcx0110/REI-Bench](https://github.com/jcx0110/REI-Bench)
  - Useful for pragmatic ambiguity taxonomies and underspecified instruction transformations.
  - Lower priority because it is simulator-grounded and not wearable / web-native.

- [xandery-geek/SuperGlasses](https://github.com/xandery-geek/SuperGlasses)
  - Useful for smart-glasses metadata schema, especially `dynamism` and tool-annotation fields.

- [tonyliao-meta/WearVQA](https://huggingface.co/datasets/tonyliao-meta/WearVQA)
  - Dataset-only official release, useful for wearable-specific image degradations and judge-based QA evaluation.
  - Lower priority because it has no interactive loop and no separate official code repo surfaced in this search.

- [ChenYi99/EgoPlan](https://github.com/ChenYi99/EgoPlan)
  - Useful for next-action planning labels and egocentric progress-to-plan formulation.

- [egoschema/EgoSchema](https://github.com/egoschema/EgoSchema)
  - Useful for long-form egocentric memory / temporal-reasoning stress tests.

- [facebook/wearable-ai](https://huggingface.co/datasets/facebook/wearable-ai)
  - Gated official dataset package rather than a normal GitHub repo.
  - Worth requesting access if EgoCUA needs a proactive wearable baseline or evaluation transfer target; expect large storage needs (`340 GB`) and non-commercial license constraints.

### Not currently cloneable from accessible official sources

- [EgoBench paper](https://arxiv.org/abs/2605.27820)
  - High-value execution neighbor, but during this search I did not find an accessible official public repo or dataset release to clone.

- [WearVQA dataset](https://huggingface.co/datasets/tonyliao-meta/WearVQA)
  - Official public artifact is the dataset page itself; no separate official code repo surfaced in this search.

### Read-only system reference

- [YoungSeng/Egocentric-Co-Pilot](https://github.com/YoungSeng/Egocentric-Co-Pilot)
  - Useful for runtime architecture and proactive clarifier design.
  - Not a benchmark, and the repo has no clear license, so treat it as read-only reference.
