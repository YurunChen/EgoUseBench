<h1 align="center"><img src="website/static/egousebench-logo.png" width="46" alt="">&nbsp; EgoUseBench</h1>

<p align="center"><strong>Benchmarking Computer-Use Agents under Evolving First-Person Contexts</strong></p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a> ·
  <a href="https://arxiv.org/abs/2611.34466">Paper</a> · <a href="https://yurunchen.github.io/egousebench/">Website</a> · <a href="https://huggingface.co/datasets/yurun-chen/EgoUseBench">Dataset</a> ·
  <a href="#quick-start">Quick start</a> · <a href="https://huggingface.co/papers/2611.34466">Hugging Face Paper</a>
</p>

## News

- **2026-11-25:** We released the [EgoUseBench evaluation code](egocua/) and [benchmark task definitions](data/tasks/).

## Abstract

AI wearables, such as smart glasses, extend the context available to agents beyond digital environments to users' everyday physical surroundings. With these devices, computer tasks can originate from users' ongoing real-world activities rather than solely from explicit instructions. Through first-person observations, agents can identify objects and activities relevant to user requests and ground digital tasks in the surrounding context. We call this interaction paradigm **Ego Use**. However, first-person observations are limited by the wearer's viewpoint and may provide incomplete or ambiguous evidence of ongoing activities. As new observations arrive, agents may need to revise their understanding of the physical context and user intent, and reassess whether prior digital actions remain valid. Whether current agents can reliably handle these changes remains insufficiently evaluated.

We introduce **EgoUseBench**, a benchmark for evaluating Ego Use across three interaction settings with different levels of first-person context. Computer-Use relies on explicit instructions, Offline Ego Use provides prerecorded first-person context before execution, and Online Ego Use provides continuous observations as user intent evolves. Based on over 74 hours of real-world first-person recordings, we construct 693 tasks across the three interaction settings. To capture task evolution, we track first-person observations, user requirements, and website state separately. We further group tasks by how information changes during execution, resulting in three types: Graft (revised user requirements), Refresh (new first-person observations), and Clarify (agent-initiated information seeking). We evaluate several advanced models and find that average success rates in Online Ego Use are 26.3% lower than in Computer-Use. This result reveals the difficulty of keeping task execution aligned with evolving first-person context and user intent.

## EgoUseBench in 70 seconds

A request can change mid-task, a new view can change which product is right, and an ambiguous request may need a question first. Real first-person footage and agent runs show how EgoUseBench tests these three changes and compares explicit instructions, video provided upfront, and observations delivered during execution.

https://github.com/user-attachments/assets/ecda234c-6fe5-43e3-93a7-13bc1a9bb022

## Quick start

With [uv](https://docs.astral.sh/uv/) and Python 3.11+, you can check the task definitions and open the project site without data or model credentials:

```bash
uv sync
uv run egocua validate data/tasks
uv run python -m website
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050). Set `WEBSITE_PORT` to use a different port.

**Run an evaluation.** You'll need the matching task media, Chrome, and an OpenAI-compatible endpoint. See the [source registry](configs/datasets.json) for access terms and media licenses. Copy `.env.example` to `.env` and set `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

```bash
uv run python -m egocua.study_cli --model <model> --export experiments/
uv run python -m egocua.study_cli --study-from experiments/<frozen-dir> --run-all
uv run egocua inspect-run <run-dir> --view score
```

The default budget is **20 steps** for a single-change task and **30** for a composite task. For local tests, run `uv run pytest -q`; run `uv run playwright install chromium` once before browser tests.

## Benchmark

**200 matched tasks** run in three settings. Online Ego Use adds **93 composite tasks**, for **693 task instances** built from **74+ hours** of source recordings.

| Setting | What the agent receives | Instances |
| --- | --- | ---: |
| **Computer-Use** | Explicit final requirements, without first-person footage | 200 |
| **Offline Ego Use** | First-person footage before execution, plus the user's requests | 200 |
| **Online Ego Use** | First-person observations and user requests as execution unfolds | 293 |

The online tasks cover **Graft** (the user updates the request), **Refresh** (a new view changes the evidence), and **Clarify** (the agent asks which object the user means). Agents operate on **EgoMart** for shopping, **EgoMedia** for saving tutorials, and **EgoMaps** for bookmarking places.

### How a task runs

[![Excerpt from the benchmark design figure: first-person observations, user requests, and the shopping cart evolve on separate tracks before the final-state check.](assets/egousebench-method.png)](assets/egousebench-method.png)

In this Graft example, the user asks for peanut butter partway through the task. The tortillas stay in the cart. The checker tests whether the final cart satisfies both requests.

## Repository layout

| Path | Contents |
| --- | --- |
| [`egocua/`](egocua/) | Timeline, agent adapters, execution loop, and final-state checker |
| [`sites/`](sites/) | Three self-hosted sites and their frozen catalogs |
| [`data/tasks/`](data/tasks/) · [`data/experiments/`](data/experiments/) | Compiled task definitions, composite tasks, and evaluation materials |
| [`configs/`](configs/) | Evaluation settings and source registry |
| [`tools/`](tools/) · [`tests/`](tests/) | Build, audit, and rendering scripts; offline tests |
| [`website/`](website/) | Project-site source |

## Acknowledgments

We thank the teams behind Ego2Web, HD-EPIC, CaptainCook4D, EgoOops, HoloAssist, EgoLife, and EgoSchema / Ego4D for the source materials, and the independent reviewers who helped check the benchmark tasks.

The explainer uses cropped, muted excerpts from [HD-EPIC (Damen et al., University of Bristol)](https://doi.org/10.5523/bris.3cqb5b81wk2dc2379fx1mrxh47) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and from [HoloAssist (Wang et al., Microsoft)](https://holoassist.github.io/) under [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/). Its narration and music are generated for this project.

## Contact

For benchmark questions or research collaboration, email [Yurun Chen](mailto:yurunchen.research@gmail.com).
