# EgoCUA-Bench 科学协议审计与修订稿

**状态：本文保留协议审计结论，不定义当前任务规模或审核队列。当前候选以 `data/selection.json` 为准，发布状态以 `data/releases/` 为准。**  
**初始审计：2026-08-05；最近更新：2026-08-10**  
**审计范围：任务设计、任务完成验证、指标与统计设计**

## 0. 结论

EgoCUA-Bench 的三轨方向值得保留，但不能把项目自定义分类写成社区共识，也不能让指标名称超出实际证据。当前 v1 已按这一原则收窄主张；真实任务是否成立仍取决于后续人工门禁。

本修订稿建议保留 Graft、Refresh、Clarify 三个诊断轨道，同时作六项强制调整：

1. 将“三源完备分类”降为本项目提出并待证伪的工作分类，不作为社区共识或已证明定理。
2. 每条任务冻结“自然语言目标—形式目标谓词—决定性证据”三元组，最终状态而非规定动作路径决定任务是否成功。
3. 将运行有效性、目标成功、轨道行为约束和证据充分性分层判定，避免把基础设施错误计为 Agent 失败。
4. 删除由被测模型结果决定的 capability filter；可解性只由任务发布前的人类基线和环境检查决定。
5. 以任务为统计单位，replicate 只估计任务内随机性；条件差使用任务内配对效应，置信区间按任务聚类重采样。
6. 把不可直接观测的心理状态词改为行为词：不把最终选择叫“目标识别准确率”，不把加购物车延迟叫“信念更新延迟”，不把非空提交叫“网页执行成功”。

Pilot 的用途应限定为：验证协议、发现地板/天花板效应、估计任务方差、校准后续正式样本量。它不用于支持稳定的模型总排名或跨开放世界场景的强泛化结论。

---

## 1. 哪些是共识，哪些不是

### 1.1 成熟共识

以下做法已在多个 computer-use / web-agent 基准中重复出现，可作为本项目的强设计依据：

- 使用可复现环境中的最终后端状态或可执行 evaluator 判断任务完成，而不是只依赖语言模型判断。代表工作包括 WebArena、OSWorld、WorkArena/BrowserGym 和 τ-bench。
- 允许多条合法操作路径，只验证任务声明要求的结果；只有当行为约束本身属于任务要求时，才检查过程。
- 冻结任务、环境版本、evaluator 和 agent 配置，并保留完整轨迹和最终状态用于复核。
- 比较条件时保持任务、环境、预算和模型配置不变，使用同一任务上的配对对照。
- 在小样本、多任务评测中报告不确定性与任务级分布，而不是只报告一个平均数。
- 发布前进行人类可解性检查和 evaluator 的人工回放审计。

### 1.2 新兴但值得采用的做法

以下设计由 2025–2026 年的新基准或审计工作支持，但仍应写成“采用的新兴规范”，不能宣称为全领域定论：

- 对评分证据使用 `PASS / FAIL / UNKNOWN`，并把 UNKNOWN 视为证据不足而不是 Agent 失败。
- Clarify 同时测量是否该问、问的字段是否有效、何时问、得到回答后是否利用；使用 Ask-F1 抑制“逢任务必问”。
- 将用户打扰视为成本，报告不同成本系数下的敏感性，而不是只优化成功率。
- 对动态任务报告事件后的 success-by-budget 曲线，而不是仅用单个平均延迟。
- 对 evaluator 的假阴性和假阳性进行分层抽检。

### 1.3 本项目选择

以下设计可以使用，但必须作为本项目的范围或工程选择，并通过 pilot 校准：

- Graft / Refresh / Clarify 三轨分类及“goal-belief divergence”理论语言。
- Clarify 最多一次提问、恰好两个合理候选和五类 slot。
- 所有任务共享同一冻结 EgoMart 商品库；构造阶段可使用目标与难度分层的候选短名单，但这些标签不进入运行时站点或 Agent 输入。
- 首次获得非空搜索结果后的下一步触发，未触发则固定第 8 步释放。
- 40 步上限和每条件 3 次 replicate。
- 精选关键帧作为上游感知模块的 oracle 输出。

---

## 2. 构念与主张边界

### 2.1 总构念

建议把总体研究问题表述为：

> 当执行中的 Agent 收到会改变最终目标解释或其充分信息的新输入时，它能否调整后续网页行为并达到更新后的可验证目标状态？

这是可观测的行为问题。基准可以讨论 Agent 的“目标维护能力”，但不应根据点击、选择或购物车动作直接断言 Agent 内部形成了何种信念。

### 2.2 三轨的严格操作定义

| 轨道 | 可支持的构念 | 处理变量 | 不应直接声称 |
|---|---|---|---|
| Graft | 指令在不同到达时机下的组合执行稳健性 | B 在开始时或执行中到达 | 所有类型的意图变化、完整的 addition/revision/retraction 能力 |
| Refresh | 新视觉证据到达后，网页行为对时间变化所指的适应与抗干扰能力 | V1 不提供、中途提供或开始提供 | 连续视频理解、主动注意、潜在信念更新速度 |
| Clarify | 在行动语境中判断是否需要澄清、提出有效问题并利用答案的能力 | 区分信息直接给出、可交互获得或不可获得 | 通用多轮对话能力或真实用户满意度 |

当前 C/P/R 是 Constraint、Parallel、Relational 三种**目标组合关系**，不是 InterruptBench 的 addition、revision、retraction 三种**更新操作**。建议保留 C/P/R，但删除“与 addition/revision/retraction 显式对齐”的表述。本 pilot 的 Graft 本质上主要研究新增信息后的组合执行；revision 和 retraction 应作为未来扩展，或另行构造真正改变/撤销旧目标的任务。

---

## 3. 统一任务契约

每个基础任务在生成条件版本之前，必须冻结以下三元组：

```yaml
task_claim: 用户完成任务后，现实中应成立的自然语言结果
goal_predicate: 对环境最终状态执行的确定性布尔谓词
decisive_artifacts: 足以让第三方重放并复核谓词的状态与日志
```

### 3.1 条件配对不变量

同一基础任务的所有条件必须保持：

- 商品目录、商品 ID、页面实现和初始后台状态一致；
- 最终目标谓词一致，除非条件本身明确定义了不同的实验目标；
- 模型、system prompt、agent harness、工具、步数/时间预算一致；
- 视觉分辨率、网页截图方式和网页可操作能力一致；
- 除被研究的信息及其到达时机外，不增加额外提示；
- 事件内容、触发规则和送达日志可机械核验。

不得通过改变页面难度、候选数量或任务措辞质量来制造条件差。

### 3.2 任务接纳门槛

任务进入冻结 pilot 前必须全部通过：

1. **目标唯一/可接受集合完备**：两名独立标注者同意所有合法最终状态均已列入 `acceptable_sets`。
2. **人类可解**：至少两名未参与该任务构造的人，在完全相同的输入通道和预算下独立完成；pilot 建议接纳阈值为两人均成功，正式版再依据 pilot 估计采用不少于 80%–90% 的人类成功门槛。
3. **条件隔离**：审阅者只能从处理变量区分条件，不能从目录、目标页面或提示风格猜出条件。
4. **视觉承重**：在需要视觉接地的任务中，移除视觉证据后，独立人类不能稳定识别唯一目标；提供完整证据后，一致识别目标。
5. **环境可执行**：目标商品存在，搜索、详情页、购物车和提交链路在冻结环境中均可完成。
6. **evaluator 充分**：正确替代路径通过，部分完成、额外商品、错误数量、错误变体和过早提交失败。
7. **无泄漏**：文件名、packet ID、商品 ID、alt text 和隐藏元数据不泄漏目标或条件。

人类验证失败、标注者分歧或证据不足的样本应修订或移出 pilot，不得在看到模型结果后选择性保留。

---

## 4. 分轨任务修订

### 4.1 Graft

#### 任务定义

- `a_only`：只执行 A，其目标谓词为 `G_A`。
- `bundled`：开始时同时收到 A 和 B，目标谓词为 `G_AB`。
- `injected`：开始收到 A，至少完成一个真实 Agent 动作后收到相同 B，目标谓词仍为 `G_AB`。

`bundled` 与 `injected` 的最终含义必须相同，B 必须是必要信息：若忽略 B，至少存在一个满足 `G_A` 但不满足 `G_AB` 的结果。否则任务不能测量注入后的目标调整。

`a_only` 不是 `bundled/injected` 的同目标反事实条件，而是基础能力和 A 阶段行为的诊断条件。核心因果比较只在 `bundled` 与 `injected` 之间进行。

#### C/P/R 接纳规则

- C（Constraint）：`Solutions(G_AB)` 是 `Solutions(G_A)` 的真子集。
- P（Parallel）：`G_AB = G_A AND G_B`，且 A、B 的完成均可由最终状态分别验证。
- R（Relational）：B 通过与 A 的关系确定目标；关系必须在视觉证据或冻结目录中可验证，不能依赖常识猜测。

#### 注入时机

主协议可继续使用进度里程碑，以提高内部可比性，但需要：

- 在首个 Agent 动作后才允许注入；
- 在提交前强制送达；
- 记录 `event_step`、归一化进度和触发原因；
- 将 wall-clock / 不同进度位置作为敏感性实验，而不是混入主分数。

### 4.2 Refresh

Refresh 的 retain 和 replace 测量方向相反，不能在一个未分层平均数中解释。

#### retain

- V0 唯一确定目标 A；V1 出现视觉上合理但任务上无关的变化。
- `stale`、`stream`、`oracle_latest` 的正确最终目标均为 A。
- V1 不得包含会合理改变用户目标的新意图信息。

#### replace

- V0 支持暂定对象 A；V1 显示自然事件使最终所指变为 B。
- `stale` 缺少确定 B 所必需的信息；`stream` 中途得到 V1；`oracle_latest` 开始即得到 V0+V1。
- `stream` 与 `oracle_latest` 的最终目标谓词完全相同，仅信息到达时间不同。

#### 必需的反事实人工检查

- 对 replace，V0-only 人类不应能以高于设计基线的水平稳定确定 B；看到 V0+V1 后应高一致率确定 B。
- 对 retain，看到 V1 的人类仍应高一致率维持 A。
- V1 不得只靠 packet 名称、时间标签或文本说明泄漏 A/B。

当前 `update_latency` 应改名为 **Post-update Target-Action Latency**，定义为 V1 送达后首次对最终正确目标采取预声明动作的步数。它是行为反应时间，不是信念更新时间。未产生正确目标动作的运行必须作为右删失/预算耗尽报告，不能从均值中静默删除。主报告优先给出 `SR(k)`：事件后 k 步内达到最终正确状态或完成预声明目标动作的任务比例。这组响应指标只在 `replace` 子集计算；把 `retain` 纳入分母会把“维持原目标”误写成对目标替换的响应。

### 4.3 Clarify

每个 v1 ambiguous 任务必须恰好有两个在初始信息下同等合理的候选，且一个合法 slot 的标准回答足以唯一化目标。

#### 三条件

- `oracle`：区分属性在初始指令中给出，ask 工具仍可用；用于测不必要提问。
- `interactive`：缺少区分属性，可问一次并获得确定性回答。
- `forced_execute`：与 interactive 输入相同但无 ask 工具；用于估计无澄清猜测基线。

#### 必需的人工检查

- 未看到隐藏答案的两名标注者都认为两个候选同样合理，且不能从候选顺序稳定猜出目标。
- 看到 canonical answer 后，两人都能唯一确定同一候选。
- 隐藏目标在候选位置、slot 和答案形式上平衡；报告实际多数类和固定位置猜测基线，而不是默认声称 chance 为 25%。
- oracle 指令自然、信息量与 canonical answer 等价，不得额外简化搜索词或页面操作。

一次提问上限作为用户成本控制的项目选择保留。需要区分四个行为阶段：

1. 是否问；
2. slot 是否有效且问题是否针对阻塞歧义；
3. 是否在首次不可逆目标动作（本环境为 submit）前问；
4. 获得回答后是否选择正确目标并成功提交。

---

## 5. 任务完成验证

### 5.1 四层判定

每次运行输出以下彼此独立的字段：

| 字段 | 含义 | 处理方式 |
|---|---|---|
| `run_valid` | 所需输入/事件送达、环境与日志完整、可评分 | false 时分类原因；基础设施故障重跑，不计 Agent 失败 |
| `goal_pass` | 最终后台状态命中一个完整 acceptable set | headline success 的核心 |
| `policy_pass` | 任务明示的过程约束成立 | 仅在轨道任务确实要求时使用，不普遍限制路径 |
| `evidence_status` | PASS / FAIL / UNKNOWN | UNKNOWN 表示保留证据不足，不等于 Agent 失败 |

建议的 headline 定义（分母只含 `run_valid=true` 的运行；基础设施无效运行按预注册规则重跑并另报）：

```text
E2E Success = goal_pass AND required_policy_pass | run_valid = true
```

对普通购物任务，`required_policy_pass` 默认为 true；不要因为 Agent 采用了不同但合法的搜索和导航路径而判失败。

若正式数据中存在 UNKNOWN，报告证据支持的成功率区间：

```text
lower = PASS / N
upper = (PASS + UNKNOWN) / N
```

同时单独报告 UNKNOWN 率和原因。设计目标应是通过完整后端状态与日志使 UNKNOWN 接近零。

### 5.2 名称修订

| 当前名称 | 问题 | 建议名称 |
|---|---|---|
| `target_correct` / 目标准确率 | 最终状态混合了视觉接地、搜索和执行 | `goal_state_correct` / 最终目标状态正确率 |
| `web_success` / 网页执行成功率 | 当前只是服务端记录了非空 submit | `submission_completed` / 有效提交完成率 |
| `update_latency` | 容易被解释为潜在信念变化 | `post_update_target_action_latency` |
| `first_target_retained` | 只看最终集合中有 A，未必代表执行中保持 | `final_state_contains_A`，或增加明确的阶段性状态证据 |

如果论文需要独立声称“视觉目标识别准确率”，必须增加不改变网页状态的预注册 grounding probe，并说明 probe 可能改变后续行为。更稳妥的主协议是只报告 outcome 指标，把 opened/carted 当作执行诊断，不把它们解释为纯感知能力。

### 5.3 决定性证据

每次运行至少保存：

- task、catalog、视觉 packet、environment、evaluator 的内容哈希；
- model、provider、版本、agent harness、prompt、工具 schema 和预算；
- 初始后台状态与最终后台状态；
- 每个事件的预定触发、实际触发、送达步数和 payload 哈希；
- 完整网页动作、ask 调用、模型响应和错误；
- evaluator 输入、接受集合、逐字段比较结果和最终判定；
- 基础设施错误与 Agent 错误的分类。

### 5.4 evaluator 对抗测试

冻结前必须至少覆盖：

- 正确目标的不同合法搜索/导航路径；
- 错误目标但访问过正确页面；
- 正确商品仅加入购物车但未提交；
- 正确目标加额外商品；
- 重复数量、错误变体、空车提交；
- Graft 的 B 未送达或提交早于送达；
- Refresh 的 V1 未送达、重复送达或顺序错误；
- Clarify 的非法 slot、第二次提问、答案在提交后到达；
- 未知商品 ID、缺失最终状态和日志截断。

发布前从 PASS 与 FAIL 中分层抽样进行人工轨迹回放，分别估计 evaluator 假阳性率和假阴性率。只审计失败样本不能证明 evaluator 没有假阳性。

---

## 6. 指标与估计对象

### 6.1 公共指标

| 指标 | 定义 | 地位 |
|---|---|---|
| Valid Run Rate | `mean(run_valid)`，同时按无效原因分解 | 协议质量，不是 Agent 能力 |
| E2E Success | 有效运行中满足最终目标及必要策略约束的比例 | 公共主结果 |
| Goal-State Correctness | 有效运行中最终选择集合命中 acceptable set | 结果诊断 |
| Submission Completion | 有效、非空且被后端接受的提交比例 | 执行诊断 |
| Correct Product Opened/Carted | 轨迹中是否访问/加入最终正确目标 | 中间诊断，不作为纯感知指标 |
| Repeat Action Rate | 预声明动作签名的重复数 / 对应动作数 | 效率诊断 |
| Steps on Success | 只在 E2E 成功运行中计算的步骤分布 | 效率诊断 |
| Cost per Verified Success | 总步骤或 token / 成功任务数 | 资源诊断，需同时报告失败率 |

不能把三个轨道混成一个总 leaderboard 分数。按 track 和 subtype 等权宏平均；任务先在 replicate 内聚合，再对任务求均值。

### 6.2 Graft

主估计量：

```text
Interruption Cost_t
  = mean_r E2E(t, bundled, r) - mean_r E2E(t, injected, r)

Track estimate = mean_t Interruption Cost_t
```

正值表示中途注入造成损失。若 provider 提供可控且跨条件配对的随机 seed,可同时报告 replicate-level S/S、S/F、F/S、F/F；若随机性不可控,不得伪配对单次运行,应报告每任务的 bundled/injected 成功率及其差值分布。

次指标：

- `Injected Integration Success`：injected 条件的 E2E；
- `Final State Contains A`：仅适用于 P 等最终状态应保留 A 的子型；
- `Post-event SR(k)`：注入后 k 步内完成 `G_AB` 的比例；
- 成功条件下的额外步骤和 token。

不得把 `final_state_contains_A` 跨 C/P/R 不加区分地解释为 First-Target Retention，因为某些正确的 revision/relation 任务本来就不要求最终保留 A。

### 6.3 Refresh

retain 与 replace 各有一个预注册主结果，不跨子型合并：

```text
Replace Adaptation Benefit_t
  = mean_r E2E(t, stream, r) - mean_r E2E(t, stale, r)

Retain Online Robustness Loss_t
  = mean_r E2E(t, oracle_latest, r) - mean_r E2E(t, stream, r)
```

对 replace 另报：

```text
Online Refresh Cost_t
  = mean_r E2E(t, oracle_latest, r) - mean_r E2E(t, stream, r)
```

对 retain 另报可观测的 `Final False-Update Outcome Rate`：stream 条件最终提交了非 A 的非空集合。它只能说明最终行为被干扰，不能证明 Agent 的内部目标信念发生过错误更新。

动态诊断使用 `Post-update SR(k)` 和右删失的 target-action latency 分布。若只给成功运行的平均 latency，必须明确标注 success-conditioned，并同时给未响应比例。

### 6.4 Clarify

Answer Utilization 只在 interactive 中“有效 slot 且在首次候选承诺动作前获得回答”的运行上计算，并检查回答后的首次候选特异动作是否指向隐藏目标。后续纠错可以保留 E2E 成功，但不能补回回答利用成功。

主估计量：

```text
Clarification Gain_t
  = mean_r E2E(t, interactive, r) - mean_r E2E(t, forced_execute, r)
```

关键策略诊断：

- Ask Recall = ambiguous interactive 任务中提问的比例；
- Ask Precision = 在每个任务都包含一个 oracle 与一个 interactive 运行的平衡评测集中，所有发生提问的运行里确实属于 ambiguous interactive 的比例；
- Ask-F1 = Ask Precision 与 Ask Recall 的调和平均；
- Valid Slot Rate = interactive 中，在已提问运行里合法 slot 的比例；
- Timely Ask Rate = 已提问运行中，在首次候选承诺动作前得到回答的比例；
- Answer Utilization = 已收到合法答案的运行中 E2E 成功的比例；
- Oracle Unnecessary Ask Rate = oracle 条件中的提问率。

为呈现用户负担，报告预声明成本网格上的敏感性，而不选择一个最有利的系数：

```text
U_lambda = E2E - lambda * Ask
lambda in {0, 0.02, 0.05, 0.10}
```

该 utility 只用于敏感性分析，不作为跨模型单一排名分数。真实用户成本需要后续用户研究校准。

---

## 7. 统计分析计划

### 7.1 分析单位

- `task_id` 是独立抽样和不确定性的主要单位。
- 模型多次运行称为 `replicate_id`，除非 provider 明确实现可控随机 seed；只记录但不能控制生成的 seed 不应被称为实验 seed。
- 先在每个 task × arm 内对 replicate 求均值，再计算任务内实验臂差，最后对任务求宏平均。

### 7.2 置信区间

- 使用任务聚类 bootstrap，重采样时保留该任务的全部条件和 replicate。
- 在 track 内按预注册 subtype 分层重采样；不能只按 track 分层，因为 C/P/R、retain/replace 的样本量和构念不同。
- 报告 95% bootstrap CI、任务级点图和配对四象限。
- pilot 中置信区间宽是正常结果，不用事后过滤任务或添加 replicate 来追求显著性。

### 7.3 重复运行可靠性

每条件报告：

- `pass@1`：replicate 的平均成功率；
- 成功次数分布（0/3、1/3、2/3、3/3）；
- 可选 `pass^3`：三次全部成功的任务比例，用作严格可靠性诊断。

`pass^3` 不替代 pass@1，也不能在不同模型使用不同 replicate 数时直接比较。

### 7.4 多重比较与主次结果

- 每轨预注册上述主估计量；其他指标明确标为 secondary/diagnostic。
- 不构造跨三轨综合总分。
- 当前 pilot 不做确认性显著性检验，也不据此排名模型；只报告预注册效应量、任务级区间和描述性可靠性。
- 正式实验样本量在 pilot 后根据任务级配对效应方差和期望 CI 宽度确定，不按 pilot 是否显著决定。

### 7.5 不使用结果驱动的 capability filter

任何“简单条件至少成功一次才纳入”的过滤器都会使不同模型在不同任务集合上被比较，并产生选择偏差。当前实现已删除这一做法：

- headline 使用全部预先通过人类可解性门槛的冻结任务；
- 基础能力问题通过条件四象限、oracle/a_only 绝对成功率和人类基线解释；
- 如需子集分析，只能使用看模型结果之前定义的任务属性或人类难度分层。

---

## 8. Pilot 验收与冻结流程

### 阶段 A：逐任务静态审计

- 两名构造标注者完成目标、证据、候选与条件不变量；
- 一名未参与构造的 evaluator 审阅者验证谓词和 acceptable sets；
- schema、资源哈希和无泄漏检查通过。

### 阶段 B：人类执行

- 每轨至少手工执行 1 条端到端 smoke test；
- 再由至少两名独立人类在正式 harness 上运行全部条件；信息充分条件必须成功，故意缺信息的控制条件只要求协议有效并完成提交；
- 记录成功、时间、错误和分歧，不只记录最终标签。

### 阶段 C：evaluator 审计

- 运行对抗单元测试；
- 从预跑 PASS 和 FAIL 各分层抽样回放；
- 报告假阳性、假阴性、UNKNOWN 和基础设施失败率；
- 修订后重新冻结 task/environment/evaluator 三者的联合版本。

### 阶段 D：模型 pilot

- 先审核 `data/selection.json` 中的当前候选，再根据逐题证据决定是否扩展。共享同一上游长视频的任务不得作为独立样本重复计数。只有每个报告类别至少一条通过人工、资产、evaluator 和真实模型门禁后，才能声称该类别已覆盖；
- 只评估流程可行性、效应方向、方差、地板/天花板和日志完备性；
- 在查看模型对照效果后不删除“表现不好”的合法任务；只有预注册的任务无效条件可以排除，并保留排除日志。

### 阶段 E：正式版预注册

- 冻结主张、主指标、排除规则、样本量、模型/agent 配置和分析脚本；
- pilot 任务可进入正式集，但正式报告需区分用于开发的任务与完全保留的测试任务；
- 给任务、环境、evaluator、prompts 和数据资产单独版本号及联合 release hash。

---

## 9. 实现映射与剩余人工工作

下表 P0/P1 代码项已并入当前 v1。剩余阻塞是候选任务尚未完成双人独立审核和独立人类执行；没有通过人工准入的任务时不启动正式模型验证。

| 优先级 | 当前实现 | 科学问题 | 修订要求 |
|---|---|---|---|
| P0 | `web_success` 取 submit 布尔值 | 名称超出判定含义 | 改为 `submission_completed`；E2E 由有效运行和目标谓词决定 |
| P0 | 无独立 `run_valid/evidence_status` | 基础设施故障可能与 Agent 失败混合 | 增加分层判定、原因枚举和 UNKNOWN 证据状态 |
| P0 | capability filter 依赖至少一次模型成功 | 结果后选择、跨模型集合不一致 | 从 headline 和默认报告中删除；改用预先人类门禁 |
| P0 | `_difference` 按 task × seed 直接配对 | seed 可能只是记录值 | 改名 replicate；先 task-arm 聚合后计算任务差 |
| P0 | bootstrap 只按 track 分层 | 子型构念与数量不同 | 按 track × subtype 分层、任务聚类重采样 |
| P1 | `target_correct` 称目标准确率 | 混合感知、搜索与执行 | 改名 `goal_state_correct` |
| P1 | Graft retention 对全部子型统一 | 某些正确目标不要求保留 A | 仅在形式目标要求 A 的任务计算，或改为描述性字段 |
| P1 | Refresh latency 忽略无正确动作运行 | 成功者偏差且暗示潜在信念 | 改行为名称，报告删失和 SR(k) |
| P1 | Clarify 缺 Ask Precision/F1 与 timing | 不能区分“会判断何时问”和“总是问” | 增加 ask policy、timing、answer utilization 指标 |
| P1 | 难度由固定干扰物数量定义 | 数量不保证知觉难度 | 用属性距离、人类混淆和 pilot 成功率校准 |
| P2 | 文档称三源完备、不可裁剪 | 属理论主张而非共识 | 改为 scoped working taxonomy，并给出排除范围 |

当前实现顺序为：判定语义与日志 → evaluator 测试 → 人类门禁与任务再审 → 根据审核后的任务方差和统计功效决定是否扩展。项目不再按预设配额凑题；任务数量不能替代逐条构念、来源和可解性门禁。

---

## 10. 主要依据

- [WebArena](https://webarena.dev/)：自托管可复现网页环境与功能正确性评价。
- [VisualWebArena](https://arxiv.org/abs/2401.13649)：视觉接地的网页任务与可执行环境。
- [OSWorld](https://arxiv.org/abs/2404.07972)：真实计算机环境中的 execution-based evaluator。
- [WorkArena](https://arxiv.org/abs/2403.07718) 与 [BrowserGym](https://arxiv.org/abs/2412.05467)：可复现浏览器任务、统一 harness 和 evaluator。
- [τ-bench](https://arxiv.org/abs/2406.12045)：以数据库最终状态验证工具 Agent，并使用 pass-k 风格的多次运行可靠性指标。
- [InterruptBench](https://arxiv.org/html/2604.00892v1)：同任务配对的中断条件、更新必要性、最终目标判定、行为四象限和事件后预算曲线。
- [Ego2Web](https://arxiv.org/abs/2603.22529)：第一人称视觉到网页执行任务，以及人类检查视觉接地、网页可执行性和指令质量的流程。
- [EgoBench](https://arxiv.org/abs/2605.27820)：过程与结果联合验证的 egocentric agent 评测。
- [STT-Arena](https://arxiv.org/abs/2605.18548)：动态触发任务中的 stale-state、误诊断和缺失验证问题。
- [CLAMBER](https://arxiv.org/abs/2405.12063)：歧义识别与澄清问题质量的分离评估。
- [HiL-Bench](https://arxiv.org/abs/2604.09408)：是否应询问的 precision/recall 与 Ask-F1。
- [Ask Early, Ask Late, Ask Right](https://arxiv.org/abs/2605.07937)：澄清时机是独立于是否提问的评测维度。
- [The Value of Information](https://arxiv.org/abs/2601.06407)：以成功收益和用户认知成本共同分析主动询问。
- [Evidence-Supported Benchmarking](https://arxiv.org/html/2605.10448)：任务—evaluator—证据三元组、PASS/FAIL/UNKNOWN 与证据支持区间。
- [How Benchmarks Mis-Score Computer-Using Agents](https://arxiv.org/html/2607.28367)：evaluator 人工回放、联合版本化和假阴性审计；该研究主要审计失败轨迹，因此本协议额外要求抽检 PASS 以覆盖假阳性。
- [RLiable](https://arxiv.org/abs/2108.13264)：小样本、多任务评测中的区间估计、分层 bootstrap 和结果分布报告。
- [Paired Bootstrap Resampling](https://aclanthology.org/W04-3250/)：成对系统比较中的配对重采样原则。

## 11. 明确不纳入本轮的内容

- 不修改 Browser-Use 或极简 loop 的内部规划算法；
- 不把安全、隐私、多人说话、音频 ASR、跨设备接力加入当前三轨；
- 不用自由生成用户模拟器替代 Clarify 的确定性回答；
- 不因追求更高分而调整任务、删除合法困难样本或改变 evaluator；
- 不在协议确认前改动现有代码和 pilot 数据。
