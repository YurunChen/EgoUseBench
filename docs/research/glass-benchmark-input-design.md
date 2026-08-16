# Glass / Egocentric Assistant Benchmark 输入设计与来源

_Last checked: 2026-08-09._

本笔记只使用官方论文、项目页、数据卡，以及 `reference/` 下保存的官方源码快照。核心判断标准只有两条：

- `dataset contains`：数据集/任务定义里客观存在什么原始素材、标注、上下文。
- `model receives`：官方评测脚本或任务协议里，模型在单个评测单位上真正能看到什么字段/模态。

如果官方只公开到任务概念、没有公开字段级 schema 或代码，我会明确写“官方未公开到字段级”；必要的地方会写“依据官方描述推断”。

## 先给结论

- 对 EgoCUA 最有参考价值的不是单个 benchmark，而是三类输入协议的组合：
  - `Ego2Web / VisualWebArena` 的“执行时网页观察 + 明确任务目标”；
  - `TeleEgo / EGOSTREAM / Wearable-AI` 的“严格流式、不可看未来、按时间释放信息”；
  - `WAGIBench / EgoPoint / REI-Bench` 的“目标不完全显式、指代不充分、需要保留歧义和候选解释”。
- 公开 benchmark 的共同缺口是：很少同时把“第一人称世界观察、网页/工具状态、用户历史、可澄清歧义、未来信息屏蔽”放进同一个输入协议里。
- 因此 EgoCUA 的关键不是继续增加模态，而是明确区分四样东西：原始数据、实际释放给 Agent 的观察、仅供评测使用的隐藏标注，以及信息释放的时间。
- `candidate target / final target / valid clarification slot / should_ask` 之类字段不能出现在 Agent 输入里，否则会泄露答案。它们只能留在任务标注和 checker 中。

| 输入协议 | 代表 benchmark | Agent 实际看到什么 | 信息如何到达 | 主要限制 |
| --- | --- | --- | --- | --- |
| 静态图像 QA | EgoPoint、SUPERGLASSES、WearVQA | 图像 + 问题（有时含选项） | 一次性 | 不测试视频时序或执行 |
| 离线视频 QA / 规划 | SAW-Bench、EgoPlan、EgoSchema | 整段视频的采样帧/视频 + 问题 | 一次性，通常可看到完整片段 | 不能据此声称具备在线更新能力 |
| 静态多模态上下文 | WAGIBench | 视频帧 + 音频转写 + 数字状态 + 历史文本 | 固定 context window 一次性给出 | 测目标推断，不测执行后的修订 |
| 严格流式 | TeleEgo、EgoProactive、EGOSTREAM | 当前及过去视频/音频/对话 | 按秒、按 chunk 或按问题时刻释放 | 没有网页执行 |
| 交互执行 | Ego2Web、EgoBench、VisualWebArena | 初始任务/世界证据 + 每步网页或工具观察 | 初始证据预给，执行状态逐步返回 | 第一人称证据通常不是持续更新的 |

## 主 benchmark

### 1. Ego2Web

官方来源：
- 论文: https://arxiv.org/abs/2603.22529
- 项目页: https://ego2web.github.io/
- 官方代码: https://github.com/Yui010206/Ego2Web
- 数据集: https://huggingface.co/datasets/Shoubin/Ego2Web

- 原始素材来源/采集设备：
  - 视频来自 Ego4D 的真实第一人称视频。
  - Ego2Web 沿用了源视频，没有重新录制智能眼镜数据；论文没有把某一种眼镜硬件作为任务前提。
- 源标注来源：
  - 官方生成管线写明先做视频 caption，再由 LLM 生成 web task，最后经人工 verification and refinement。
  - `reference/Ego2Web/ego2web_generation/README.md` 与 `data_generation.py` 说明了“caption -> instruction generation -> human verification”三步。
- 评测单位：
  - 一个 `video-instruction pair`。
  - 官方公开数据 `ego2web.json` 中每条样本最少包含 `vid / instruction / domain / visual_clues`。
- `dataset contains`：
  - 第一人称视频；
  - 自然语言 web 任务指令；
  - 领域标签；
  - 视觉线索时间段 `visual_clues`。
- `model receives`：
  - 这里必须区分三层：
    - 任务生成输入：caption 管线看到的是整段第一人称视频，被转成详细 caption 后再生成 task instruction。
    - agent 实际输入：并不统一，取决于被评测 agent。
    - judge 输入：轨迹、截图、任务指令与视频证据会再送入 `Ego2WebJudge`。
  - 官方代码与论文设定并非“所有模型都直接吃原视频”：
    - Gemini 类模型可直接接收原视频；
    - GPT-4.1 / GPT-4o 使用关键帧；
    - Claude / GPT-5.4 computer-use 之类 browser agent 在公开实现里是接收由 Gemini 先抽出的结构化 caption/视觉摘要，再去做网页执行。
  - 本地官方 `Ego2WebJudge` 实现对非 Gemini judge 默认在标注时间段内均匀抽取 8 帧。这个数值能作为固定视觉预算的工程参考，但它是 judge 端设置，不是所有 Ego2Web Agent 共同遵守的输入协议，也没有证明 8 帧在科研上最优。
  - 因此 Ego2Web 的 `dataset contains` 是整段视频，但 `model receives` 在不同 agent 之间并不一致。
  - 在执行阶段，模型还会看到网页观察与自身轨迹；项目页明确其评测是 `web execution`，不是纯离线 QA。
  - 官方公开样例 JSON 本身不包含网页观察；这些是执行环境运行时提供的，不是静态标注文件字段。
- 信息是离线一次性还是按时间释放：
  - 物理世界证据对大多数公开实现是一次性预给的，不是 streaming assistant 协议。
  - 即使底层视频被抽成关键帧或 caption，也是在网页执行前整体准备好。
  - 网页观察按交互步逐步释放。
- 是否包含未来信息/历史：
  - 物理世界部分是整段视频离线给定，因此相对“流式 assistant”设定，默认包含该视频后续帧。
  - 网页执行部分只包含当前及过去轨迹，不包含未来网页状态。
- 任务指令如何产生：
  - 官方生成脚本要求任务必须依赖视频里的至少两个视觉 cue，并限制到常见公开网站域名；然后人工验收。
- 是否另有网页/工具观察：
  - 有，且这是它相对多数 wearable benchmark 的关键差异：除了第一人称视频，还有真实网页执行观察。

对 EgoCUA 的直接启发：
- 它证明了“world-grounded evidence -> web execution”可测。
- 但它把物理世界证据做成整段离线输入，不足以测试 `Refresh` 式在线替换与 `Clarify` 式主动澄清。

### 2. EgoBench

官方来源：
- 论文: https://arxiv.org/abs/2605.27820

- 原始素材来源/采集设备：
  - 论文报告共 `1,045 egocentric-video-grounded tasks`，覆盖四个 daily scenarios。
  - 其中 82.4% 使用自行采集的智能眼镜/手机第一人称视频，17.6% 使用 Ego4D；视频片段主要为 10--30 秒。
  - 官方代码/项目页在本次核查中未公开可访问，因此不能进一步核验字段名与设备型号。
- 源标注来源：
  - 官方摘要明确采用 `three-stage synergistic pipeline` 设计任务。
  - 可确认有 user-agent-tool interactive environment 与 simulated user。
- 评测单位：
  - 一个以第一人称视频为锚点的 interactive tool-using task。
- `dataset contains`：
  - 可以确认包含：第一人称视频锚点、任务、工具环境、用户交互环境、确定性验证框架。
  - 官方未公开到字段级 schema。
- `model receives`：
  - 可以确认模型会收到：第一人称视频上下文、用户动态反馈、工具调用结果、环境状态更新。
  - 这是根据官方摘要中的 `user-agent-tool interactive environment` 推断到协议级；字段名与 JSON 结构未公开。
- 信息是离线一次性还是按时间释放：
  - 不是一次性静态 QA。
  - 用户反馈与工具结果按交互轮次释放。
- 是否包含未来信息/历史：
  - 设计目标显然是不包含未来工具结果，而是让代理在动态交互中获取。
  - 但因为官方代码未公开，无法核实视频本体是否整段离线提供、还是按片段给定。
- 任务指令如何产生：
  - 官方摘要只确认三阶段设计流程；没有公开到 prompt/schema 级。
- 是否另有网页/工具观察：
  - 有工具观察，但不是公开网页 benchmark；它是结构化 tool environment。

对 EgoCUA 的直接启发：
- 它最像 `Graft`：用户信息会动态补充，且工具调用改变状态。
- 但它不是 live web，也缺少当前可复核的公开输入 schema；因此更适合做“设计邻居”，不适合直接复刻。

### 3. WAGIBench

官方来源：
- 论文 / OpenReview: https://openreview.net/forum?id=REG4cJItSZ
- 项目页: https://facebookresearch.github.io/WAGIBench/
- 官方代码: https://github.com/facebookresearch/WAGIBench

- 原始素材来源/采集设备：
  - 项目页明确：29 小时多模态数据，348 participants，3477 recordings。
  - 任务围绕 assistive wearable agent goal inference；采集材料是 wearable 场景第一人称上下文。
  - 官方公开页未进一步点名具体眼镜硬件。
- 源标注来源：
  - 官方项目页明确每条样本有 `ground-truth goals`，并伴随 visual / audio / digital / longitudinal contextual observations。
- 评测单位：
  - 一条 goal inference 样本。
  - MCQ 设置里是“给视频和四个 digital-action 选项，输出 A/B/C/D”。
  - Generative 设置里是“输出一个 digital action dictionary”。
- `dataset contains`：
  - 视频；
  - transcription；
  - digital state；
  - longitudinal history；
  - 参考 goal / 选项 goal。
  - 官方 README 列出的数据文件包括 `ob2_v2.4.4_1_mcq.json`、`ob2_transcriptions_250501.json`、`digital_state_v2.4.4.json`、`structured_goals_longitudinal_history_v244.json`。
- `model receives`：
  - 官方 prompt 与构造代码表明，模型收到的是：
    - 视频帧；
    - 当前样本转写文本（若有）；
    - digital state 文本化摘要；
    - longitudinal history 中的过去动作/世界状态文本；
    - MCQ 选项或 generative 模板。
  - 关键点是：公开代码里的 `audio` 通常不是 raw audio 直接喂模型，而是以 transcription 形式进入 prompt。
  - 这些不是原样 JSON 直喂，而是经过 prompt builder 串成固定 `context window` 文本上下文。
- 信息是离线一次性还是按时间释放：
  - 离线一次性给定一个 context window。
  - 没有在线时间推进。
- 是否包含未来信息/历史：
  - 包含显式历史：`longitudinal_history`。
  - 不做“未来屏蔽”式 streaming；它测的是从静态多模态上下文猜当前 latent goal。
- 任务指令如何产生：
  - 不是用户口述指令，而是以 ground-truth digital goal 为监督，要求模型预测用户“接下来想做的数字动作”。
- 是否另有网页/工具观察：
  - 没有 live web。
  - 有 digital app state 观察，但仍是静态上下文，不是运行时工具反馈。

对 EgoCUA 的直接启发：
- 很适合 `Graft` 前半段：在用户未明确说出口前，先维护多个 intent hypothesis。
- 但它没有 commit 之后的 revision loop；EgoCUA 需要把这种 latent-goal 表示接到后续执行上。

### 4. Wearable-AI / EgoProactive

官方来源：
- 数据卡: https://huggingface.co/datasets/facebook/wearable-ai
- Leaderboard / 官方评测说明: https://huggingface.co/spaces/facebook/wearable-ai-leaderboard
- 论文: https://arxiv.org/abs/2606.04970

- 原始素材来源/采集设备：
  - 数据卡定义为 `egocentric (first-person, head-mounted wearable camera) videos`。
  - 验证集三条任务各 700 样本，总计 2100 视频；数据卡写明全量约 317 GB。
- 源标注来源：
  - 官方数据卡把任务拆成 `egoconv / egolongqa / egoproactive` 三个配置。
  - EgoProactive 的监督是每个 candidate moment 上“说/不说”以及说什么。
- 评测单位：
  - `EgoProactive`：一个 streaming session，内部按 8 秒 chunk 顺序预测。
  - 官方 leaderboard 文案明确：每个 8-second video chunk 输出一个动作。
- `dataset contains`：
  - 视频；
  - 三个任务配置的标注文件；
  - starter kit / evaluation scripts。
- `model receives`：
  - 对 `EgoProactive`，模型按序只看当前及过去视频/上下文。
  - 输出必须是 `$interrupt$...` 或字面量 `$silent$`。
  - 对 `EgoConv`，模型还会看到多轮对话，并把自己此前回答当上下文。
  - 另一个容易误写的点：公开 challenge 数据的视频是 `audio-stripped`，所以公开评测协议不是“原始音视频 assistant”，而是去音轨的视频协议。
- 信息是离线一次性还是按时间释放：
  - `EgoProactive` 是严格按时间 chunk 释放。
  - `EgoConv` 也是 turn-based progressive release。
- 是否包含未来信息/历史：
  - 官方数据卡和 leaderboard 明确写 `no future leaking`。
  - 只允许使用当前 turn/chunk 之前的视频与上下文。
- 任务指令如何产生：
  - `EgoProactive` 不是用户明确发问，而是 benchmark 指定候选时刻，要求模型决定是否应主动插话以及插什么话。
- 是否另有网页/工具观察：
  - 没有网页/外部工具执行观察。

对 EgoCUA 的直接启发：
- 这是 `Graft` 的时间策略模板：先判断“现在值不值得说”，再决定说什么。
- EgoCUA 可直接吸收它的 no-future contract 和 chunk-level speaking policy。

### 5. TeleEgo

官方来源：
- 论文: https://arxiv.org/abs/2510.23981
- 项目页: https://teleai-uagi.github.io/TeleEgo/
- 官方代码: https://github.com/TeleAI-UAGI/TeleEgo
- 数据集: https://huggingface.co/datasets/David0219/TeleEgo

- 原始素材来源/采集设备：
  - 官方项目页与 README：真实用户连续日常活动记录，5 位参与者，3 天/人，约 14.4h/人。
  - 场景覆盖 work/study/social/shopping/health/travel 等。
- 源标注来源：
  - 官方 README：专家/人工构建 QA，覆盖 3 个大维度 12 个子类。
  - 每题都有 evidence timestamp。
- 评测单位：
  - 一条在时间轴某个时刻触发的问题。
  - 若首次答对，还会在 60s、120s ... 后追加 recall 问题，用于算 MPT。
- `dataset contains`：
  - 合并长视频；
  - timeline 映射；
  - QA JSON；
  - narration / speech / event description 等多模态注释。
- `model receives`：
  - 官方 harness 明确是逐秒释放，但不同模型的单秒输入形式不一样：
    - Qwen2.5-Omni：`1s frame/raw audio`；
    - Gemini 类实现：`1s video clip`；
    - GPT-4o 等伪流式实现：通常是 `1 frame + ASR/transcript`。
  - 到 `evidence end` 对应时刻才把文本问题附加到当前输入单元；
  - 系统消息还可带当前子视频段 description；
  - 如果启用 history，则保留过去对话；不启用时就是严格当前流输入。
- 信息是离线一次性还是按时间释放：
  - 严格 streaming。
  - 逐秒送入，问题在对应秒点触发。
- 是否包含未来信息/历史：
  - 不包含未来。
  - 允许只通过模型内部/显式对话 history 保留过去。
  - 题目与 recall 都只基于到当前时刻为止已见内容。
- 任务指令如何产生：
  - 人工构造 question-answer，对准时间证据点。
- 是否另有网页/工具观察：
  - 没有网页/工具执行观察。

对 EgoCUA 的直接启发：
- 这是 `Refresh` 最强邻居：它把“什么时候该忘、什么时候该保留”做成了显式协议。
- EgoCUA 的世界状态缓存应模仿它，把每条记忆都挂在时间轴和有效期上，而不是做无结构长上下文拼接。

### 6. EgoPoint-Bench

官方来源：
- 论文: https://arxiv.org/abs/2604.21461
- 项目页: https://guyyyug.github.io/EgoPoint-Bench/
- 官方代码: https://github.com/GUYYYUG/EgoPoint
- 数据集: https://huggingface.co/datasets/GUYYYUG/EgoPoint

- 原始素材来源/采集设备：
  - 项目页与 README：模拟数据 `Point-Sim` + 真实第一人称采集。
  - README 把真实部分标成 `MLVision Capture`，共 1162 real samples。
- 源标注来源：
  - capability-oriented QA annotation；
  - 覆盖 5 个能力维度与 3 级 deixis。
- 评测单位：
  - 一条 pointing-grounded QA 样本。
- `dataset contains`：
  - 图像；
  - 问题；
  - 选项或答案；
  - 维度标签、deixis level、难例属性。
  - 本地 real 样例 JSON 还包含 `rationale / relevance_score / type` 等。
- `model receives`：
  - 官方 benchmark 本质上是单图 QA。
  - 样例字段与 README 都表明，模型看到的是第一人称图像 + 问题；若是 MCQ，还会看到选项。
  - pointing 信息嵌在图像本身，而不是额外坐标字段。
- 信息是离线一次性还是按时间释放：
  - 离线一次性。
- 是否包含未来信息/历史：
  - 不含流式历史。
- 任务指令如何产生：
  - 由 capability taxonomy 和 deixis 层级系统化构造。
- 是否另有网页/工具观察：
  - 没有。

对 EgoCUA 的直接启发：
- 这是 `Clarify` 的最干净参照：不要把“指代解析”当成普通视觉 QA。
- EgoCUA 的隐藏标注应显式保留 referent 候选集和唯一可消歧属性；Agent 只能从原始视觉和指令自行判断是否需要提问。

### 7. SUPERGLASSES

官方来源：
- 论文: https://arxiv.org/abs/2602.22683
- 官方代码: https://github.com/xandery-geek/SuperGlasses
- 数据集: https://huggingface.co/datasets/xandery/SuperGlasses

- 原始素材来源/采集设备：
  - README 与论文明确：全部是 real-world data collected entirely by smart glasses devices。
  - 论文列出三种设备：Ray-Ban Meta、Xiaomi、RayNeo V3。
- 源标注来源：
  - 参与者先采集真实场景中的 image-query pair；人工标注者再补充答案、难度、推理步骤、聚焦区域、拆分后的子问题，以及图像/文本检索轨迹。
  - 官方 queries 配置提供 question/answer、`sub_questions` 与 retrieval tool annotation。
- 评测单位：
  - 一条 image-question pair。
- `dataset contains`：
  - 去重后的眼镜图片；
  - 问题、答案；
  - `glasses / domain / category / language / dynamism / difficulty / location / hops / use_image / sub_questions / image_quality` 等。
- `model receives`：
  - 官方 `main.py` 与 `dataset.py` 表明模型实际吃到的是：
    - 图像；
    - 问题；
    - 其余字段主要用于切片分析和结果记录，不直接进基础 VQA 调用。
  - 官方 baseline 会从 `sub_questions` 提取 retrieval tool 标记，但默认 VQA 推理仍是 `question + image -> prediction`。
- 信息是离线一次性还是按时间释放：
  - 离线一次性。
- 是否包含未来信息/历史：
  - 无历史，无 streaming。
- 任务指令如何产生：
  - 就是人工标注问题；README 未公开 prompt 生成流程。
- 是否另有网页/工具观察：
  - 没有真实网页观察；
  - 但元数据显式记录了“理论上需要的 retrieval tools”，这对 EgoCUA 很有价值。

对 EgoCUA 的直接启发：
- `sub_questions + retrieval-tools` 说明静态 QA benchmark 也能标“如果要答好，需要哪类外部检索”。
- EgoCUA 可以把“需要哪类外部检索”作为分析标注，但不能直接告诉 Agent 本轮应调用什么工具。

### 8. WearVQA

官方来源：
- 论文: https://arxiv.org/abs/2511.22154
- 数据卡: https://huggingface.co/datasets/tonyliao-meta/WearVQA

- 原始素材来源/采集设备：
  - 官方论文说明图片由付费参与者使用 Ray-Ban Meta 智能眼镜采集，场景来自日常生活；旁观者经像素化处理。
  - 采集者被要求同时覆盖自然出现和有意覆盖的模糊、遮挡、弱光、未放大、裁切/倾斜等眼镜画质问题。
- 源标注来源：
  - 数据卡 viewer 暴露出 question/answer 与多种视觉质量和手部属性标签。
- 评测单位：
  - 一条 image-question 样本。
- `dataset contains`：
  - 图片；
  - 问题、答案；
  - `domain`；
  - `hand_finger_elements`；
  - `is_blur / is_cut_off / is_low_light / is_occluded / is_not_zoomed_in` 等质量标签；
  - `question_type` 等附加字段。
- `model receives`：
  - 在官方公开信息里，只能确认最基本的 `image + question`。
  - 没有公开官方评测代码可证实这些质量标签会不会进入 prompt，因此默认视为切片分析元数据，而非基础模型输入。
- 信息是离线一次性还是按时间释放：
  - 离线一次性。
- 是否包含未来信息/历史：
  - 无。
- 任务指令如何产生：
  - 图片采集后由标注流程构造问题与参考答案，并按领域、认知任务和画质问题分类；问题被约束为只依赖图像和常识即可回答。
- 是否另有网页/工具观察：
  - 没有。

对 EgoCUA 的直接启发：
- 它提醒了 wearable 画质退化本身就是输入设计问题：模糊、遮挡、过近/过暗不该只留给模型硬抗。
- EgoCUA 可以把“图像质量不确定度”做成上游字段，触发 `Clarify` 或 `Refresh`。

### 9. SAW-Bench

官方来源：
- 论文: https://arxiv.org/abs/2602.16682
- 项目页: https://sawbench.github.io/
- 官方代码: https://github.com/UCSB-AI/SAW-Bench
- 数据集: https://huggingface.co/datasets/ucsbai/SAW-Bench

- 原始素材来源/采集设备：
  - 项目页与 README：786 个真实视频，全部由 Ray-Ban Meta Gen 2 眼镜自录。
- 源标注来源：
  - 2071 条 human-annotated QA；
  - 分成 6 个 observer-centric 空间任务。
- 评测单位：
  - 一条视频 QA，多为多选题。
- `dataset contains`：
  - egocentric 视频；
  - `question / options / ground_truth / answer / key / scene_category`。
- `model receives`：
  - 官方 `construct_prompt.py` 与 `gpt.py` 表明基础协议是：
    - 从视频均匀采样帧序列；
    - 把这些帧按时间顺序喂给模型；
    - 再给多选问题和选项。
  - 项目页还明确“the model sees only the video, with no bird's-eye map”。
- 信息是离线一次性还是按时间释放：
  - 整个短视频在单轮评测中离线提供，但实际输入通常是均匀采样帧，而不是逐步 streaming。
- 是否包含未来信息/历史：
  - 相对 streaming 设定，包含整段视频对应的未来帧。
- 任务指令如何产生：
  - 人工标注 observer-centric spatial reasoning 问题。
- 是否另有网页/工具观察：
  - 没有。

对 EgoCUA 的直接启发：
- `Refresh` 不只等于“更新事实”，还包括“保持 observer-centric spatial state”。
- 如果 EgoCUA 有空间参照型任务，任务标注和评测器需要记录“观察者相对目标的位置如何变化”；这些标注属于隐藏真值，不能作为额外提示直接提供给 Agent。

### 10. EGOSTREAM

官方来源：
- 项目页: https://saroo25.github.io/Egostream/

- 原始素材来源/采集设备：
  - 官方项目页明确说它是从已有 egocentric benchmarks curated questions，来源包括 Ego4D Episodic Memory VQA、EgoLife、EgoTempo、Multi-Hop EgoQA、HD-EPIC。
  - 也即它不是新采设备，而是二次整合与重协议化。
- 源标注来源：
  - 官方页写明 `2,250 validated memory questions`，以及 `8,528 recall-conditioned evaluations`。
- 评测单位：
  - 一个 evidence-grounded memory question 在若干 recall regime 上反复评测。
- `dataset contains`：
  - 来自上游 benchmark 的 egocentric 视频证据；
  - 记忆问题；
  - Answer Validity Window (`AVW`)；
  - memory category；
  - recall regime。
- `model receives`：
  - 官方页明确：`Models process video sequentially and answer using only visual content observed up to the recall time.`
  - 也就是说，核心输入协议是“只给到当前 recall time 为止的视频观察 + 当前问题”。
  - 官方 GitHub 仍标 `coming soon`，所以字段级 schema 尚未公开。
- 信息是离线一次性还是按时间释放：
  - 严格 streaming。
- 是否包含未来信息/历史：
  - 明确不含未来；
  - 只允许使用到 recall time 为止的内容。
- 任务指令如何产生：
  - 从已有数据集题目重整为 evidence-grounded streaming memory 协议。
- 是否另有网页/工具观察：
  - 没有。

对 EgoCUA 的直接启发：
- `AVW` 是它最值得借走的概念：不是所有历史事实都该永久保留。
- EgoCUA 的 `Refresh` 最好给每条外界事实一个 `valid_from / valid_to / stale_after_observation`。

## 邻接对照

### 11. VisualWebArena

官方来源：
- 论文: https://arxiv.org/abs/2401.13649
- 项目页: https://jykoh.com/vwa
- 官方代码: https://github.com/web-arena-x/visualwebarena

- 原始素材来源/采集设备：
  - 不是第一人称世界数据，而是自建网站环境与网页任务配置。
- 源标注来源：
  - 官方 `config_files/vwa/*.raw.json` 直接把任务 intent、起始 URL、登录要求、reference answer / evaluator program 写成结构化配置。
- 评测单位：
  - 一条网页任务配置。
- `dataset contains`：
  - 任务目标；
  - 起始 URL；
  - 所需站点；
  - 登录态；
  - evaluator 配置；
  - difficulty 标签。
- `model receives`：
  - 官方 prompt JSON 显示模型实际收到：
    - 用户 objective；
    - 当前网页截图或 SoM 标注图，或 accessibility tree；
    - 当前 URL；
    - open tabs；
    - previous action。
  - 这些是运行时网页观察，不是静态数据集字段。
- 信息是离线一次性还是按时间释放：
  - objective 一次性给定；
  - 网页观察按步释放。
- 是否包含未来信息/历史：
  - 不含未来网页状态；
  - 只带过去 action history。
- 任务指令如何产生：
  - 由任务模板实例化。
- 是否另有网页/工具观察：
  - 有，而且这是 benchmark 核心。

对 EgoCUA 的意义：
- 它是 Ego2Web 的网页执行邻居。
- EgoCUA 应直接继承它那种“world state 与 browser state 分层输入”的做法，而不是把网页也压成纯文本结果。

### 12. REI-Bench

官方来源：
- 论文 / OpenReview: https://openreview.net/forum?id=vmBIF25KLf
- 项目页: https://jcx0110.github.io/rei-bench-project/
- 官方代码: https://github.com/jcx0110/REI-Bench

- 原始素材来源/采集设备：
  - 基于 ALFRED / AI2-THOR，而非真实 wearable 采集。
- 源标注来源：
  - 官方 repo 通过 REI split 文件把原本明确的任务重写成 vague / underspecified / coreferential 指令。
  - 本地 `data/rei_bench/splits/rei_bench.json` 可见显式 dialogue memory 与 final pending instruction。
- 评测单位：
  - 一个 embodied planning task。
- `dataset contains`：
  - ALFRED 轨迹；
  - REI 重写后的历史对话/记忆；
  - 模糊最终指令；
  - difficulty 条件。
- `model receives`：
  - 官方 evaluator 会把 `Human previous inquiry` 与 `Human pending instruction` 组织成输入；
  - 同时 agent 在 AI2-THOR 中接收 egocentric RGB observation。
  - 这是“对话记忆 + 当前模糊指令 + embodied observation”的组合。
- 信息是离线一次性还是按时间释放：
  - 历史对话一次性附带；
  - 环境观察按执行步释放。
- 是否包含未来信息/历史：
  - 包含显式历史对话；
  - 不包含未来环境状态。
- 任务指令如何产生：
  - 通过系统化重写，制造 referential/coreferential vagueness。
- 是否另有网页/工具观察：
  - 没有网页；
  - 有 embodied 环境观察。

对 EgoCUA 的意义：
- 它说明 `Clarify` 不一定非要由显式疑问句触发。任务标注应指出哪些指代表达存在欠指明，以及什么属性能唯一消歧；这些信息只供数据校验和评测使用，Agent 仍只能从自然指令与视觉证据中自行判断是否需要追问。

### 13. EgoPlan

官方来源：
- 论文: https://arxiv.org/abs/2312.06722
- 项目页: https://chenyi99.github.io/ego_plan/
- 官方代码: https://github.com/ChenYi99/EgoPlan

- 原始素材来源/采集设备：
  - 由 Epic-Kitchens-100 与 Ego4D 构成。
- 源标注来源：
  - 官方 README：human-verified evaluation data。
  - 每条样本有 task goal、问题、多选项、当前观察帧与 task progress metadata。
- 评测单位：
  - 一道“下一步该做什么”的多选题。
- `dataset contains`：
  - video source / video id；
  - task goal；
  - question；
  - 四个选项；
  - `current_observation_frame`；
  - `task_progress_metadata`。
- `model receives`：
  - 官方 baseline 并不是把全部原始 JSON 文本直接喂入。
  - `video_llama_interface.py` 会抽取从任务开始到当前帧的 clip，再配 question + 四个答案候选。
  - `egoplan_video_llama_interface.py` 进一步把最近 `n_actions` 的动作片段和当前图像分开编码。
- 信息是离线一次性还是按时间释放：
  - 离线一次性给定一个“到当前帧为止”的 planning snapshot。
- 是否包含未来信息/历史：
  - 显式包含过去的 `task_progress_metadata`；
  - 不包含当前帧之后的未来动作。
- 任务指令如何产生：
  - 官方问题模板固定为“Considering the progress shown in the video and my current observation in the last frame, what action should I take next...”
- 是否另有网页/工具观察：
  - 没有。

对 EgoCUA 的意义：
- 它证明“把历史执行压成最近动作摘要 + 当前观察”是可行的。
- 但 EgoCUA 需要再加一层：历史里哪些事实已失效，哪些需刷新。

### 14. EgoSchema

官方来源：
- 论文: https://arxiv.org/abs/2308.09126
- 项目页: https://egoschema.github.io/
- 官方代码: https://github.com/egoschema/EgoSchema

- 原始素材来源/采集设备：
  - 长视频第一人称数据；官方 repo 通过 `questions.json` 与视频下载脚本组织。
  - 设备细节主要承接 Ego4D 生态，官方 repo未在首页重述采集硬件。
- 源标注来源：
  - 官方只公开全部视频与问题；
  - 只公开 500 条 `subset_answers.json` 供离线实验，其余通过服务器/Kaggle 评测。
- 评测单位：
  - 一道五选一长视频理解题。
- `dataset contains`：
  - 视频；
  - `questions.json`；
  - 小规模公开答案；
  - 提交/验证接口。
- `model receives`：
  - 官方 benchmark 定义上是长视频 + 问题 + 5 个选项。
  - 各官方 baseline 会先把视频抽帧再做 option scoring；如 `mPLUG-Owl` 基线把问题和每个选项组合成 yes/no 判别。
- 信息是离线一次性还是按时间释放：
  - 离线一次性长视频 QA。
- 是否包含未来信息/历史：
  - 包含整段长视频，因此对 streaming assistant 而言默认有未来帧。
- 任务指令如何产生：
  - 官方 repo未在首页公开生成流程细节。
- 是否另有网页/工具观察：
  - 没有。

对 EgoCUA 的意义：
- 它适合参考长视频抽帧与长时间记忆负载，但不提供在线事件释放或网页执行协议。

## 对 EgoCUA `Graft / Refresh / Clarify` 的具体建议

### A. 先分清三种数据可见性

EgoCUA 不需要修改 Browser-Use 内部规划逻辑，但 harness 必须把字段分成三类：

1. `source provenance`：原视频、时间段、抽帧规则、指令或答案的标注来源。用于复现，不直接喂给 Agent。
2. `released observation`：当前时刻允许 Agent 看到的指令、关键帧、澄清回答和网页观察。
3. `hidden ground truth`：最终目标、候选实体、有效澄清槽位、事件触发点和 checker 条件。只供环境与评测器使用。

这种拆分比把所有字段拼进一个大 prompt 更重要。尤其不能把 `hidden_target_id`、`final_target_id`、`valid_slot` 或 `clarification_needed` 暴露给 Agent。

### B. `Graft`：只在事件时释放新指令

参考来源是 WAGIBench 的目标不完全显式、EgoProactive 的 no-future contract，以及 EgoBench 的轮次化用户反馈。

- `a_only` 只在开始释放 A；`bundled` 开始释放 A+B；`injected` 开始只释放 A，到指定浏览器事件再释放 B。
- A、B 的文本来源、与源视频的证据时间段、注入触发原因应写入 provenance/hidden annotation，不作为额外提示发给 Agent。
- headline 比较中三种条件除“B 的到达时机”外必须完全一致，包括关键帧、网站状态、候选目录和 action budget。

### C. `Refresh`：按包释放真实新视觉证据

参考来源是 TeleEgo 的逐时刻释放、EGOSTREAM 的 no-future/有效期思想，以及 SAW-Bench 的连续第一人称状态变化。

- `V0` 与 `V1` 必须来自同一源视频中按时间先后出现的真实片段；任务文件保存各自 timestamp 和 frame extraction rule。
- pilot 固定总图像预算为 8：V0、V1 各 4 张，并在各自连续区间内包含端点均匀采样。不同区间允许得到不同等效 Hz，避免为了固定 fps 而改变每题图像数。
- `stream` 在触发前只能看到 `V0`，触发后才收到 `V1`；`stale` 永远不收到 `V1`；`oracle_latest` 开始时收到两包。
- `retain / replace`、`final_target_id` 和目标变化类型是隐藏评测标注，不能随视觉包发给 Agent。
- 第一轮 pilot 不必给每条事实建内部状态缓存；只需确保事件日志能回答“Agent 在哪一步收到哪一包信息”。

### D. `Clarify`：给工具，不给歧义答案

参考来源是 EgoPoint 的指代歧义和 REI-Bench 的欠指明语言。

- Agent 初始只看到自然指令和相同的关键帧；`candidate_ids / hidden_target_id / valid_slot / canonical_answer` 全部隐藏。
- `interactive` 仅多出一次 `ask_user(slot, question)` 能力；标准答案只有在合法调用后才释放。
- `oracle` 直接在自然语言指令里加入区分属性，但仍保留工具，用来测不必要提问；`forced_execute` 与 interactive 输入相同，只移除工具。
- 三条件之间不能额外提供候选列表、置信度或“此题有歧义”提示，否则 Clarify 会被简化为按提示调用工具。

### E. 建议公开的最小任务与事件记录

建议公开的是输入来源与释放记录，而不是替 Agent 计算好的世界状态：

```json
{
  "episode_id": "string",
  "track": "graft | refresh | clarify",
  "source": {
    "dataset": "Ego2Web/Ego4D",
    "video_id": "string",
    "spans": [{"packet_id": "V0", "start_s": 10.0, "end_s": 14.0}],
    "frame_extraction": {"strategy": "uniform_inclusive_endpoints", "count": 4}
  },
  "events": [
    {
      "event_id": "E0",
      "type": "instruction_injection | visual_context_update | clarification_answer",
      "trigger": "episode_start | first_search_results | max_step_8 | valid_tool_call",
      "payload_ref": "V1 or instruction_B",
      "released_at_step": 5
    }
  ],
  "hidden_evaluation": {
    "final_target_id": "not exposed to agent",
    "valid_slot": "not exposed to agent"
  }
}
```

运行日志还应逐步记录 Agent 实际收到的消息、关键帧、网页观察和工具返回。这样可以审计 no-future、条件配对和失败归因，而不需要改 Agent 的内部记忆表示。

## 最终判断

- `Ego2Web` 和 `VisualWebArena` 告诉我们：执行型 benchmark 必须把网页观察单独建模。
- `TeleEgo` 和 `EGOSTREAM` 告诉我们：assistant benchmark 不能默认给整段未来视频，必须有严格时间协议。
- `WAGIBench`、`EgoPoint-Bench`、`REI-Bench` 告诉我们：输入设计里必须容纳“未明说目标”和“未消歧指代”。

因此，EgoCUA 的输入设计最应该新增的不是更多模态，而是三件事：

1. `No-future, time-indexed input contract`
2. `Strict separation between released observations and hidden evaluation fields`
3. `Auditable event/provenance log`

没有这三件事，`Graft / Refresh / Clarify` 最终都会退化成普通多模态问答。

## Sources and References

1. Ego2Web project: https://ego2web.github.io/
2. Ego2Web paper: https://arxiv.org/abs/2603.22529
3. Ego2Web code: https://github.com/Yui010206/Ego2Web
4. Ego2Web dataset: https://huggingface.co/datasets/Shoubin/Ego2Web
5. EgoBench paper: https://arxiv.org/abs/2605.27820
6. WAGIBench project: https://facebookresearch.github.io/WAGIBench/
7. WAGIBench paper: https://openreview.net/forum?id=REG4cJItSZ
8. WAGIBench code: https://github.com/facebookresearch/WAGIBench
9. Wearable-AI dataset card: https://huggingface.co/datasets/facebook/wearable-ai
10. Wearable-AI leaderboard: https://huggingface.co/spaces/facebook/wearable-ai-leaderboard
11. Wearable-AI / EgoProactive paper: https://arxiv.org/abs/2606.04970
12. TeleEgo project: https://teleai-uagi.github.io/TeleEgo/
13. TeleEgo paper: https://arxiv.org/abs/2510.23981
14. TeleEgo code: https://github.com/TeleAI-UAGI/TeleEgo
15. TeleEgo dataset: https://huggingface.co/datasets/David0219/TeleEgo
16. EgoPoint-Bench project: https://guyyyug.github.io/EgoPoint-Bench/
17. EgoPoint-Bench paper: https://arxiv.org/abs/2604.21461
18. EgoPoint-Bench code: https://github.com/GUYYYUG/EgoPoint
19. EgoPoint-Bench dataset: https://huggingface.co/datasets/GUYYYUG/EgoPoint
20. SUPERGLASSES code: https://github.com/xandery-geek/SuperGlasses
21. SUPERGLASSES paper: https://arxiv.org/abs/2602.22683
22. SUPERGLASSES dataset: https://huggingface.co/datasets/xandery/SuperGlasses
23. WearVQA paper and dataset card: https://arxiv.org/abs/2511.22154 ; https://huggingface.co/datasets/tonyliao-meta/WearVQA
24. SAW-Bench project: https://sawbench.github.io/
25. SAW-Bench paper: https://arxiv.org/abs/2602.16682
26. SAW-Bench code: https://github.com/UCSB-AI/SAW-Bench
27. SAW-Bench dataset: https://huggingface.co/datasets/ucsbai/SAW-Bench
28. EGOSTREAM project: https://saroo25.github.io/Egostream/
29. VisualWebArena project: https://jykoh.com/vwa
30. VisualWebArena paper: https://arxiv.org/abs/2401.13649
31. VisualWebArena code: https://github.com/web-arena-x/visualwebarena
32. REI-Bench project: https://jcx0110.github.io/rei-bench-project/
33. REI-Bench paper: https://openreview.net/forum?id=vmBIF25KLf
34. REI-Bench code: https://github.com/jcx0110/REI-Bench
35. EgoPlan project: https://chenyi99.github.io/ego_plan/
36. EgoPlan paper: https://arxiv.org/abs/2312.06722
37. EgoPlan code: https://github.com/ChenYi99/EgoPlan
38. EgoSchema project: https://egoschema.github.io/
39. EgoSchema paper: https://arxiv.org/abs/2308.09126
40. EgoSchema code: https://github.com/egoschema/EgoSchema
