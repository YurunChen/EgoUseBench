# Ego2Web 与当前 EgoCUA-Bench 的任务难度比较

## 结论

当前 60 条任务不能因为 scripted smoke 只用了 3–5 步，就被判断为“真实 Agent 三五步即可完成”。这个 smoke 脚本预先知道正确商品 ID，只验证事件、网页状态和 checker 是否接通，刻意绕过了视频理解、搜索词生成、候选比较和纠错。它不是模型难度实验。

与 Ego2Web 相比，当前任务的难度呈现出两个相反方向：

- **网页执行明显更简单、更可控**：当前所有任务都在同一个静态 EgoMart 网站上完成，动作基本是搜索、查看商品、加入购物车和提交。Ego2Web 使用多个真实在线网站，任务常要求跨站比较、读取多个页面属性或返回开放式信息，还会遇到网页变化、登录、CAPTCHA 和加载失败。
- **交互协议更难，也更适合做诊断**：Ego2Web 的指令和视频证据在开始时给出，当前 Graft、Refresh、Clarify 则分别要求 Agent 处理中途追加要求、后来出现的新视觉状态和一次主动澄清。这些不是 Ego2Web 原任务所测的能力。

因此，目前最准确的判断不是“当前任务都太简单”，而是：**当前 Benchmark 主动降低了网站操作的开放性，以换取动态交互变量的可控和可归因；但还没有真实模型结果证明这 60 条任务的总体难度分布合理。**

## Ego2Web 的任务是什么样的

### 指令形式和长度

官方发布的 [`ego2web.json`](../../reference/Ego2Web/dataset/ego2web.json) 含 500 条任务。对其中 `instruction` 字段按空格分词复算得到：

| 统计量 | 英文词数 |
|---|---:|
| 最短 | 13 |
| 25% 分位 | 27 |
| 中位数 | 33 |
| 均值 | 33.87 |
| 75% 分位 | 41 |
| 最长 | 63 |

指令通常由三部分构成：从视频中确定对象或事件、在指定网站执行操作、返回或验证若干结果。例如任务会要求识别“第四个拿起的零食”，再到 Amazon 报告每份热量；也会要求在 Amazon 和 eBay 找同一物品并比较价格。

当前任务的初始指令更短：Graft 的 A 平均 20.63 词，Refresh 的 base 平均 21.38 词，Clarify 的 ambiguous 平均 21.14 词；但 Graft 的 bundled 完整指令平均 30.80 词，已经接近 Ego2Web 的 33 词中位数。长度本身不是难度指标，Ego2Web 更重要的复杂度来自多属性结果和真实网站执行。

### 网页域和任务类型

官方数据文件的实际类别分布为：

| 类别 | 任务数 |
|---|---:|
| E-Commerce | 230 |
| Media Retrieval | 132 |
| Knowledge Lookup | 92 |
| Local / Maps | 31 |
| Others | 15 |

论文附录列出的站点包括 Amazon、eBay、Walmart、YouTube、Wikipedia、Google Maps、Yelp、Tripadvisor、StackExchange、Reddit、Booking 等。按官方列表对指令中的显式站点名复算，442 条指定一个站点，45 条指定两个站点，8 条指定三个站点；另有 5 条采用“major online retailer”等泛称，不能简单归入某个显式站点。

当前 60 条任务则全部固定在 EgoMart：40 条最终只需提交一件商品，20 条 Graft 任务最终提交两件商品。与 Ego2Web 相比，它的站点覆盖和网页工作流明显更窄。

## Ego2Web 的 Agent 输入和执行链

论文对任务的形式化定义是：给 Agent 第一人称视频 `V` 和任务指令 `I`，Agent 在浏览器环境中执行网页动作序列，直到形成目标状态。实际视觉输入随 Agent 能力而变化：

- Gemini 系 Agent 接收 raw video；
- GPT-4.1/GPT-4o 系 Agent 接收稀疏关键帧；
- Claude 系列和论文中的 GPT-5.4 computer-use 不能直接接收视频时，先由 Gemini-3.1-Pro 把视频转成带时间戳的详细文字描述，再交给 Agent。

Agent 随后操作真实在线网站，产生网页动作、逐步截图和最终回答。Ego2WebJudge 再读取任务指令、Agent action history、筛选后的关键网页截图以及标注时间段中的视频证据，做二分类成功判断。作者为所有 Agent 设置最多 40 步。

这与当前 EgoCUA-Bench 有两个关键区别：当前 Agent 统一接收按协议释放的关键帧包，并在固定网站中操作；Refresh 的 V1 和 Graft 的补充指令可以在第一次出现搜索结果后到达，Clarify 还允许一次确定性的 `ask_user`。

## Ego2Web 是否报告了步骤或难度

论文没有给出 500 条任务的真实平均步数、中位数、分位数或按步骤划分的 easy/medium/hard 成绩。它只明确说明：网页轨迹可能包含 **5–20 步**，其中包括加载、回退和 UI 错误等无关步骤；实验统一采用 **40 步上限**。所以不能把“5–20”当成正式的全量步骤分布。

官方仓库随 Ego2WebJudge 发布了三个可复查的轨迹示例：

| 示例 | 任务概要 | `action_history` 条数 | 网页截图数 | 结果 |
|---|---|---:|---:|---|
| 0 | Amazon 白色外套价格与折扣比较 | 21 | 20 | Agent 自报成功 |
| 1 | eBay 笔记本与黑色手柄加入购物车 | 5 | 4 | Agent 自报成功 |
| 2 | Walmart 啤酒和能量饮料价格比较 | 12 | 11 | Agent 自报失败 |

来源为官方 [`ego2webjudge/data/example`](../../reference/Ego2Web/ego2webjudge/data/example)。`action_history` 包含最后的 `done` 记录，因此不能与某些论文中只计网页动作的 step 直接等同；第一条 21 条记录也说明论文的“5–20”是概括性描述，不是已公开轨迹的严格上限。

论文更可靠的难度证据是模型成功率。人工评测成功率分别为：Claude 3.7 26.4%、Claude 4.5 32.8%、GPT-5.4 30.6%、SeeAct 34.2%、Browser-Use + GPT-4.1 44.4%、Browser-Use + Gemini-3-Flash 58.6%。在 Gemini-2.5-Pro judge 下，各模型跨域平均成功率为：Knowledge Lookup 50.0%，Media Retrieval 30.1%，Local/Maps 23.1%，E-Commerce 21.7%，Others 14.4%。作者据此认为电商和地图任务比结构化知识检索更难。

## 与当前 60 条任务逐维度比较

| 难度来源 | Ego2Web | 当前 EgoCUA-Bench | 判断 |
|---|---|---|---|
| 视频理解 | 完整视频、raw video / 关键帧 / caption 因 Agent 而异；常含序数、品牌、动作和空间线索 | 统一 8 张关键帧，Refresh 为 4+4 分时释放 | Ego2Web 视觉信息更丰富；当前输入预算更统一 |
| 网页范围 | 多个真实在线网站，开放搜索空间 | 一个固定 EgoMart，705 条唯一商品 | 当前明显更简单 |
| 网页目标 | 搜索、比较、读取多个属性、导航、媒体检索，少量购物车任务 | 全部归一为选商品、加购物车、提交 | 当前明显更窄 |
| 运行时变化 | 指令和视频通常在开始时给出 | Graft 中途改指令；Refresh 中途给新画面；Clarify 可追问 | 当前更难，且是主要研究变量 |
| 评分 | LLM judge 结合视频、轨迹和截图；开放但有判断误差 | 冻结商品 ID 与最终购物车状态确定性检查 | 当前更可复现，但目标类型更单一 |
| 已有实证 | 6 类 Agent、500 条任务成功率；上限 40 步 | 只有 scripted smoke；旧架构单例结果已经失效 | 暂时不能实证声称当前更难或更容易 |

按三条轨道看：

- **Graft**：A-only 的基础购物任务通常比典型 Ego2Web 更简单；bundled 的文字长度接近 Ego2Web，但网站仍简单；injected 的目标更新与保留使其在交互控制上更难。
- **Refresh**：单看最终网页操作很简单，但 `stream` 必须在执行中更新目标或抵抗错误更新，这是 Ego2Web 没有单独控制的难点。
- **Clarify**：网页操作简单，但 `interactive` 的核心不是搜索步数，而是识别“证据不足”、提出有效问题并利用回答；Ego2Web 没有这个能力对照。

## 对当前实验的直接建议

下一轮不应仅通过增加网页步骤来“做难”。更合理的是先用 GPT-5.4-mini 真实运行每轨代表任务，至少记录：首次搜索前的视觉识别、搜索结果候选数量、模型回合数、网页动作数、回退或重复动作、事件发生前后目标是否一致。只有这些数据才能判断任务是不是过于直白。

如果真实 Agent 大量在 4–6 回合稳定成功，再增加与研究变量相关的难度，例如更相似的视觉候选、需要打开详情页才能看到的约束属性，以及对 Graft/Refresh 事件前错误选择的撤销；不应为了模仿 Ego2Web 而直接引入多个无关网站或 CAPTCHA，因为这会削弱失败归因。

## 一手来源

- Yu et al., *Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos*, arXiv:2603.22529: https://arxiv.org/abs/2603.22529
- Ego2Web 官方仓库与数据说明：[README](../../reference/Ego2Web/README.md)
- 官方 500 条任务标注：[`dataset/ego2web.json`](../../reference/Ego2Web/dataset/ego2web.json)
- 官方任务构造代码：[`ego2web_generation/data_generation.py`](../../reference/Ego2Web/ego2web_generation/data_generation.py)
- 官方 Judge 实现与三个轨迹示例：[`ego2webjudge`](../../reference/Ego2Web/ego2webjudge)
