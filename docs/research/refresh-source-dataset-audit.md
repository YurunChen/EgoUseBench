# EgoCUA Refresh 候选连续第一人称视频数据源审计

_Last checked: 2026-08-09._

本笔记只使用官方论文、项目页、数据说明页，以及官方仓库/数据页。目标不是泛泛罗列 egocentric dataset，而是判断哪些数据源真的适合支撑 `Refresh`：也就是持续观察“用户刚才拿了什么、放回了没有、换拿了什么、当前应保留哪一个实体状态”。

核心判断标准：

- `连续视频`：是否提供未切碎的第一人称长视频，而不是仅有短 clip。
- `时序监督`：是否有动作段、叙事时间戳、对象运动段、语音/转写时间戳。
- `replace/retain 证据`：是否能从官方标注直接或较低成本地恢复“取起 -> 持有 -> 放下/放回 -> 再选择”的证据链。
- `实体识别上限`：官方标签体系最多能把对象识别到什么粒度；如果只有自由文本或 QA，没有统一对象 ontology，我会明确写弱。
- `小规模可下载性`：是否能先合法下载单序列、单 participant、单 benchmark 子集，而不是必须先吞下 TB 级全集。

## 先给结论

- 真正最适合做 `Refresh` 正样本构建的，是 `HD-EPIC` 和 `EPIC-KITCHENS-100 + VISOR`。它们都提供连续第一人称厨房视频，但 `HD-EPIC` 明确把对象运动标成 `pick-up -> placement`，还把同一物体多次移动关联起来，最接近 `replace / put-back / re-select` 所需监督。
- `Ego4D`、`Ego-Exo4D`、`AEA` 更适合做 `retain` 层或 broad-context 补充：视频连续、覆盖广、可合法分批下载，但对象“放回原处/换拿另一个”的证据大多不是现成字段，需要从 narrations、state-change、speech-to-text 或 commentary 中再挖。
- 本轮实际筛选验证了上述分工：HD-EPIC 提供了 4 条 replace 视频候选，EgoSchema/Ego4D 提供了 3 条 retain 视频候选。后者依靠“第一件”的自然时序指令保持 A，不把缺失的对象追踪标注伪装成 replace 真值。
- `EGTEA Gaze+`、`ADL`、`Charades-Ego`、`EgoLife`、`UT Ego` 都有价值，但更像高召回挖掘池或邻接对照，不是最稳的主训练源。尤其是 `Charades-Ego` 是 acted/scripted，`UT Ego` 和 `ADL` 太旧，`EgoLife` 当前公开形态更偏 assistant/QA 项目而不是现成对象替换监督。
- 在本次仅用官方来源的核查里，没有找到一个“公开、连续、可下载、且官方已给对象取放证据”的专门第一人称购物 benchmark。购物场景更现实的路线是从 `EgoLife / Ego4D / UT Ego` 这类泛日常数据中挖片段，而不是寄希望于单独 shopping dataset。

## 分级总表

| 数据源 | 分级 | 连续第一人称视频 | 官方时序信号 | `replace / retain` 证据强度 | 实体识别上限 | 下载方式 / 小规模可下载 | 访问与许可 | 典型规模 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HD-EPIC | 适合 `replace` | 是 | narration/action 段、对象运动段、gaze priming、audio 段 | 很强：官方直接标 `pick-up -> placement`，并给同物体 movement association | 很强：`558` ingredients、noun classes、`19.9K` object movements、`413` fixtures | `data.bris` 可直接浏览资源；官方 `hd-epic-downloader` 支持按部分下载；可先下单 participant/单资源 | `CC BY 4.0` | `41` 小时、`4.4M` 帧、`69` recipes |
| EPIC-KITCHENS-100 + VISOR | 适合 `replace` | 是 | action start/end、narration timestamp、VISOR masks/contact relations | 强：EPIC 给动作和叙事，VISOR 给长时手-物关系；但“同一物体多次取放”的显式关联弱于 HD-EPIC | 强：EPIC `300` noun classes；VISOR `257` object classes | 官方下载脚本、RDSF/Academic Torrents；可先下少量 participant 视频和 annotations，但不如 CLI 子集细粒度 | annotations 官方仓库为 `CC BY-NC 4.0`；商业许可需邮件申请 | `100` 小时、`700` videos、`89.9K` actions、VISOR `36` 小时/`179` untrimmed videos |
| Ego4D | 适合 `retain` | 是 | narrations、FHO、state change、PNR、tracking、benchmark clip schema | 中强：官方有 object state change / PNR / tracking，但不是天然“放回原处”标注，需要二次构图 | 中：没有一个统一厨房式 noun ontology；不同 benchmark 的对象字段粒度不同 | 官方 CLI 可下 `viz`、`narrations`、特定 benchmark、benchmark clips；非常适合先小规模试验 | 需签数据协议，AWS 凭证有效期 14 天 | `3670+` 小时；benchmark clips 约 `1TB` |
| Ego-Exo4D | 适合 `retain` | 是 | synchronized takes、transcription、expert commentary、ego/exo aligned views | 中：长视频和 commentary 很强，但官方主文档未突出统一对象放回标签 | 中弱：更偏 multiview / skill / pose / commentary，统一对象 ontology 不强 | 官方 downloader 可按 `parts / views / uids / downscaled_takes/448` 下载，小规模友好 | 需签数据协议；官方写明研究与商业使用均可申请，无费用 | `1286.3` ego+exo 视频小时（repo v2 口径），默认下载约 `12 TiB` |
| AEA (Aria Everyday Activities) | 适合 `retain` | 是 | speech-to-text、共享时钟、多用户同步、trajectory、eye gaze | 中：有时间同步 STT，可做“刚才说过/做过什么”与 retain；但缺少显式对象取放标签 | 弱到中：官方文档未给统一对象 ontology，主要是 STT + 轨迹 + MPS | 官方支持 sample sequence、单 sequence、只下 VRS 等；非常适合小规模起步 | 需接受 AEA Dataset License Agreement | `143` recordings、`7.5+` 小时、约 `353GB` |
| EGTEA Gaze+ | 仅适合高召回挖掘 | 是，但域窄且偏脚本化做饭 | frame-level action、gaze、audio、hand masks | 中：很多动作本身是 object manipulation，但“放回同一件物体”需要从动作序列再推 | 中：官方说有 hundreds of objects，但公开主标签更像动作而非统一对象 tracking | 官方网站可直接拿 raw videos、trimmed clips、gaze、annotations；小规模非常方便 | 官方网页公开下载 | `28` 小时、`86` sessions、`32` subjects |
| ADL | 仅适合高召回挖掘 | 是 | activities、object tracks、hand positions、interaction events | 中强：论文和下载页明确有 object tracks + interaction events，但规模和年代限制明显 | 中：对象级 bbox + interaction，好于纯动作集；但 ontology 和分辨率较老 | 官方页可直接下 `43.1GB` 视频与 `2.2MB` annotations | 官方论文页直链下载；未见现代化单独 license 页面 | `10` 小时、`20` 人、`20` homes、约 `1M` frames |
| Charades-Ego | 仅适合高召回挖掘 | 否，更像大量短段 acted household video | temporal annotations、text descriptions | 弱到中：有时间段，但核心是脚本化 action，不适合恢复真实“先拿错再换拿” | 弱到中：`157` action classes；对象粒度不如厨房 object-tracking 数据集 | 官方可直接下 `480p 11GB` / 原始 `47GB` / frames `53GB` | 官方项目页公开下载 | `7,860` videos、`68,536` temporal annotations |
| EgoLife | 仅适合高召回挖掘 | 是 | 长时日常录制、assistant/QA 导向标注；官方公开视频/数据卡 | 中：官方摘要明确含 shopping/cooking/socializing，但现公开说明更偏 QA/assistant，而非现成 object movement labels | 弱到中：公开对象 ontology 不清晰 | Hugging Face 数据卡公开，适合拉少量行/子目录；但官方 issue 也显示部分发布仍在补齐 | 代码仓库与数据卡均公开，具体数据使用条款需按官方页面执行 | 官方论文称约 `300` 小时；HF 卡显示约 `512GB` |
| UT Ego | 仅适合高召回挖掘 | 是 | 只有 important regions，没有动作链 | 弱：含 shopping 场景，但没有官方 object manipulation 时序监督 | 弱：重要区域而非统一对象状态 | 官方可直接下 `1.4GB`；最轻量 | 官方项目页直链下载 | `4` 个 `3-5h` 视频 |
| VISOR（单独看） | 不适合作主源 | 否，是 EPIC 的派生标注层 | masks、relations | 强于对象接触，弱于原始来源独立性；不能脱离 EPIC 视频使用 | `257` object classes | 依附 EPIC 视频 | 依附 EPIC | `36` 小时派生层 |
| EPIC-SOUNDS（单独看） | 不适合作主源 | 否，是 EPIC 音频派生层 | audio event timestamps | 只能增强“有声取放/开关/碰撞”线索，不能独立支撑 Refresh | `44` audio classes | annotations 易得，底层依赖 EPIC 视频/音频 | 官方站点公开；底层依赖 EPIC 使用条件 | `117.5K` audio events |

## 各数据源细看

### 1. HD-EPIC

官方来源：

- 项目页: https://hd-epic.github.io/
- 论文: https://arxiv.org/abs/2502.04144
- 官方标注仓库: https://github.com/hd-epic/hd-epic-annotations
- 官方下载仓库: https://github.com/hd-epic/hd-epic-downloader
- 官方数据页: https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47

为什么它最适合 `Refresh`：

- 官方 README 直接写明 object movement 标注是从 `pick-up` 到 `placement` 的时间段，而不是只给一个“拿了杯子”的动作 label。
- 同一物体在整段视频里的多次移动会被人工分组为 `association`。这正好对应 `retain` 和 `replace` 最难的地方：当前手里这个杯子是不是刚才那只、是否已经放回、是否换成另一只。
- `mask_info.json` 还把运动关联到 3D 位置、2D bbox 和 `fixture`。这让“放回原柜子/原台面”这类问题变成可监督，而不只是语言猜测。
- `eye_gaze_priming` 还给出了 pick-up 和 put-down 前的 gaze priming，这对构造“候选目标被提前看过”很有价值。

局限：

- 域很窄，几乎就是厨房与 recipe 场景。
- 规模虽然足够做高质量正样本，但广度不如 Ego4D / Ego-Exo4D。

结论：

- 如果只选一个主正样本源来做 `replace / put-back / re-select`，优先级应当最高。

### 2. EPIC-KITCHENS-100 + VISOR

官方来源：

- EPIC-KITCHENS 主页: https://epic-kitchens.github.io/
- EPIC-KITCHENS-100 页面: https://epic-kitchens.github.io/2020-100
- 官方 annotations 仓库: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- VISOR 页面: https://epic-kitchens.github.io/VISOR/
- VISOR 论文: https://arxiv.org/abs/2209.13064

为什么它依然是强主源：

- `EPIC-KITCHENS-100` 本体是连续、真实、非脚本厨房视频，且 action 段、narration、verb/noun taxonomy 都已成熟。
- `VISOR` 把其中 `179` 条未切分视频做成了长时的 hands + active objects 像素级追踪和 contact relations，显著提高“这个对象一直在不在手里/台面上”的可恢复性。
- 相比只用 EPIC action labels，`EPIC + VISOR` 更能支撑“错误 retain”和“应该 replace”的判断。

为什么还略弱于 HD-EPIC：

- 官方没有像 `HD-EPIC` 那样把每次 object movement 明确标成 `pick-up -> placement` 并做同物体 association。
- 所以很多 `put-back` 证据需要自己从 action 段、mask 序列、hand-object relation 里二次恢复。

操作建议：

- 如果要快速做第一版高质量 Refresh 数据，`EPIC-KITCHENS-100` 提供主视频与动词/名词骨架，`VISOR` 提供局部高密度时空证据，是最稳妥的双层组合。

### 3. Ego4D

官方来源：

- 项目页: https://ego4d-data.org/
- 下载文档: https://ego4d-data.org/docs/start-here/
- 更新文档: https://ego4d-data.org/docs/updates/
- Hands & Objects benchmark: https://ego4d-data.org/docs/benchmarks/hands-and-objects/
- Annotation schema: https://ego4d-data.org/docs/data/annotations-schemas/
- 官方仓库: https://github.com/facebookresearch/Ego4d
- 论文: https://arxiv.org/abs/2110.07058

对 `Refresh` 的价值：

- 覆盖场景最广，日常环境最接近真实 wearable assistant 要面对的多样世界。
- 官方明确有 `object state change classification`、`state change object detection`、`PNR`、`EgoTracks` 等任务，说明“对象何时改变、何时被重新关联”在 Ego4D 内部是正式建模对象。
- 下载机制成熟，可只先拿 `narrations`、`viz`、某个 benchmark，适合先跑小规模验证。

为什么我把它放到 `retain` 而不是 `replace`：

- 它很强，但更像一个“广谱母体”。官方对象信息分散在不同 benchmark，且没有 HD-EPIC 那样直接面向 `pick-up -> placement` 的统一 movement ontology。
- 对于 `Refresh`，它特别适合做：
  - broad-context `retain`；
  - object state change 邻接任务；
  - 长时间 re-association 负样本。
- 但若要精确监督“同一物是否被放回/换拿”，仍常需要额外构图。

### 4. Ego-Exo4D

官方来源：

- 项目页: https://ego-exo4d-data.org/
- 入门与下载: https://docs.ego-exo4d-data.org/getting-started/ 、 https://docs.ego-exo4d-data.org/download/
- 官方仓库: https://github.com/facebookresearch/Ego4d
- 论文: https://arxiv.org/abs/2311.18259

对 `Refresh` 的价值：

- 官方下载器支持按 `views ego/exo`、`uids`、`parts`、`downscaled_takes/448` 下载，非常适合先做小规模原型。
- 有 `take_transcription`、`expert_commentary`、多视角同步和长 take，对“刚才发生了什么”与后验校验很有帮助。
- 如果要做“ego 看到的物体被遮挡后，是否还能用 exo 或 commentary 验证”的研究，它比纯 ego 数据集更强。

为什么不是最优主源：

- 官方主文档更强调 multiview、skill、pose、commentary，并没有把 object placement / put-back 作为核心公开标签来卖点化。
- 因此它适合 `retain` 和 rich context，但不适合作为第一主源去直接蒸馏 `replace` 判据。

### 5. AEA（Aria Everyday Activities）

官方来源：

- AEA 页面: https://www.projectaria.com/datasets/aea
- AEA 文档: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset
- 下载文档: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_download_dataset
- 论文: https://arxiv.org/abs/2402.13349

优点：

- 官方文档明确给 `speech-to-text annotation`、共享时钟、多用户同步、trajectory、eye gaze。
- 下载器支持 sample sequence、单 sequence、只下载 VRS 等；这比很多大数据集更适合工程试验。
- 日常活动跨度比厨房数据更广，比较适合构造 `retain` 型样本，例如“刚才谁说要拿哪个物品”“上一时刻用户在同一空间做过什么”。

缺点：

- 官方没有给出统一对象 ontology，也没有明确 object movement / placement 层。
- 所以它更适合 `retain`、history grounding、speech-conditioned refresh，而不是最直接的 `replace` 监督。

### 6. EGTEA Gaze+

官方来源：

- 官方页面: https://cbs.ic.gatech.edu/fpv/
- 论文: https://arxiv.org/abs/2006.00626

优点：

- 官方页明确公开 raw videos、trimmed clips、gaze、audio、frame-level action annotations、hand masks，下载极轻量。
- 厨房 object manipulation 很密，拿来做高召回挖掘或早期原型非常划算。

缺点：

- 域窄，且带较强 recipe/script 结构。
- 对 `Refresh` 而言，它更像“有很多可挖的 manipulation”，而不是“自然日常里真的会忘、会换、会放回”的长期 assistant 场景。

### 7. ADL

官方来源：

- 论文页与下载页: https://web.cs.ucdavis.edu/~hpirsiav/papers/ADLdataset/
- 论文 PDF: https://www.cs.cmu.edu/~deva/papers/ADL_2012.pdf

优点：

- 这是少数很早就明确提供 `continuous (non-segmented) video streams`、`object tracks`、`hand positions`、`interaction events` 的第一人称 ADL 数据。
- 官方下载页仍可直接拿到 `43.1GB` 视频与 annotations。

缺点：

- 年代较早，分辨率、规模、对象 taxonomy、场景多样性都明显落后于近年的厨房/Aria/Ego4D 系列。
- 适合作为经典 object-centric 邻接对照，不适合作为现在的主训练源。

结论：

- 可用于补一些“低资源但对象交互明确”的高召回样本；不建议作为主体。

### 8. Charades-Ego

官方来源：

- 项目页: https://prior.allenai.org/projects/charades-ego
- 论文: https://arxiv.org/abs/1804.09626

优点：

- 规模不小，下载方便，有 temporal annotations 和 textual descriptions。
- 室内日常动作覆盖面不错。

缺点：

- 本质上是 crowd-sourced acted scripts；不是自然长时连续生活流。
- 对 `Refresh` 最关键的“真实对象保留/替换错误”帮助有限，更适合动作/语言粗监督。

结论：

- 可以做辅助高召回，不建议做主源。

### 9. EgoLife

官方来源：

- 项目页: https://egolife-ai.github.io/blog/
- 官方代码: https://github.com/evolvinglmms-lab/egolife
- 官方数据卡: https://huggingface.co/datasets/lmms-lab/EgoLife
- 论文: https://arxiv.org/abs/2503.03803

为什么值得看：

- 官方论文摘要明确写了 `shopping, cooking, socializing, entertainment` 等真实一周日常活动，并给出约 `300` 小时 egocentric 数据。
- 这是本次核查里最接近日常 assistant 设定的“连续生活流”之一。

为什么仍只放在高召回层：

- 目前公开说明更偏 EgoLifeQA / assistant benchmark，而不是现成 object placement supervision。
- 官方 issue 也反映部分数据/标注发布仍在补齐；因此它很有潜力，但不如 HD-EPIC / EPIC / Ego4D 那样稳定可依赖。

### 10. UT Ego

官方来源：

- 官方页: https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html

为什么单独记一笔：

- 它是本次官方核查里少数明确写出包含 `shopping` 的第一人称连续日常视频源。
- 但官方只提供 `important regions` 级别标注，不提供真正可用于 `Refresh` 的对象操作链。

结论：

- 对“购物场景存在性”有参考价值；对 `replace/retain` 监督价值很弱。

## 关于派生层：VISOR 与 EPIC-SOUNDS

这两个都不应单独当作原始主数据源：

- `VISOR` 非常有用，但它是建立在 `EPIC-KITCHENS-100` 上的 pixel-level 派生层。它增强的是“对象在哪里、是否接触、是否跨时间保持一致”，不是独立视频源。
- `EPIC-SOUNDS` 也是建立在 EPIC 音频流上的派生层。它能帮助发现碰撞、开关、倒液体、摩擦等“有声 state change”，但不能独立解决对象 identity 与 placement。

更合理的用法是：

- `EPIC-KITCHENS-100` 提供主视频和 verb/noun skeleton；
- `VISOR` 提供手-物接触与长时 mask；
- `EPIC-SOUNDS` 提供是否发生 audible event 的额外证据。

## 哪些能实际先下载“小规模片段”

最友好的几类：

- `AEA`：官方直接支持 sample sequence、单 sequence、只下 VRS。
- `Ego4D`：官方 CLI 允许只下 `viz`、`narrations`、指定 benchmark。
- `Ego-Exo4D`：官方支持按 `uids / parts / views / downscaled_takes/448` 过滤。
- `HD-EPIC`：官方数据页有分层资源与 downloader，可先按 participant / 资源类型试下。
- `EGTEA Gaze+`：官网直接提供 raw videos、trimmed clips、gaze、annotations。
- `UT Ego`：只有 `1.4GB`，直接拿来做最轻量 sanity check。
- `Charades-Ego`：`11GB` 的 480p 包已足够做小规模实验。

相对不那么友好：

- `EPIC-KITCHENS-100`：虽然官方有下载脚本和多种镜像，但工程上还是偏 bulk release。
- `ADL`：体量不大，但年代久、组织方式旧。
- `EgoLife`：数据卡公开，但如果你要稳定复现实验，还需要接受它当前公开粒度更偏 QA/assistant 发布而不是标准 object-tracking 发布的现实。

## 对 EgoCUA Refresh 的具体建议

### 1. 主正样本源：先上 `HD-EPIC`

最直接原因不是它最大，而是它最接近你要的 supervision：

- object movement 明确从 `pick-up` 标到 `placement`
- 同一物体多次移动被做了 association
- 有 fixture / 3D location，可判断“放回原处”还是“换到别处”

如果要做 `replace` 标签，HD-EPIC 能最少依赖启发式。

### 2. 第二主源：`EPIC-KITCHENS-100 + VISOR`

这组应该承担：

- 把 `replace / retain` 模式从少量超高质监督扩到更大规模
- 用 VISOR 弥补纯 action labels 看不到的手-物持续关系
- 用 EPIC noun/verb taxonomy 做更稳的对象与动作词表

### 3. `retain` / broad-context 层：`Ego4D + Ego-Exo4D + AEA`

这三者更适合补以下能力：

- 场景跨度
- 非厨房日常
- 多模态上下文
- 较长历史上的 object/state persistence

其中：

- `Ego4D` 适合 object state change 与 long-term re-association 邻接任务
- `Ego-Exo4D` 适合 commentary / multiview 辅助验证
- `AEA` 适合 speech-to-text 驱动的 retain / reminder 场景

### 4. 高召回挖掘池：`EGTEA Gaze+ / ADL / Charades-Ego / EgoLife / UT Ego`

更适合做：

- 稀有模式补样
- shopping / social daily life 邻接片段挖掘
- 早期 prompt / parser / weak label 试验

不适合做：

- 主要监督来源
- 直接定义最终标签 ontology

### 5. 现实一点的购物场景策略

在官方可核对来源内，没有发现一个已经公开到“连续第一人称 + 对象取放监督 + 易下载”的 shopping 专项 benchmark。

所以更现实的策略是：

1. 用 `HD-EPIC` 和 `EPIC/VISOR` 学 `replace / put-back` 的监督形态。
2. 用 `EgoLife`、`UT Ego`、`Ego4D` 的泛日常/购物片段，把这一判据迁移到非厨房域。
3. 不要等“理想 shopping dataset”才开始做 Refresh。

## 本轮实际下载与筛选结果

本轮没有停在数据集清单上，而是从官方源实际下载了一个小规模、可复核的片段集：

- `HD-EPIC` 全量 narration 高召回扫描：固定正则命中 87 行、33 个视频；机器结果只表示“值得回看”，不表示替换真值。
- `HD-EPIC`：7 个源视频片段。连续视频审查后保留 4 个 `video_screened` 事件，拒绝 3 个真值不唯一或发布图像证据不足的事件。
- `EgoSchema / Ego4D`：7 个官方三分钟购物或服装片段；4 个形成 `video_screened` retain 候选、2 个明确拒绝、1 个仍待核查。
- 对 8 个 Refresh 保留事件，按输入协议各抽取 `V0 4 张 + V1 4 张`，共 64 张模型实际可见的候选截图：HD-EPIC replace 32 张，EgoSchema/Ego4D retain 32 张。

| 事件 | 连续视频证据 | 预筛结果 | 原因 |
| --- | --- | --- | --- |
| 金属清洁球 → 绿黄洗碗海绵 | 两个不同 object association；A 结束后 B 被拿起、加洗洁精并继续使用 | 保留 | 后续视觉包稳定显示 B，最终实体唯一 |
| 第一块餐巾 → 灰色花纹餐巾 | A 被放回；B 有独立 association 并持续用于擦手 | 保留，但首个对象仍需人工确认可读性 | B 清楚，A 的可见时间较短 |
| 米色擦手布 → 白色红边擦手布 | 两个非重叠 object association；后者被明确选中并持续使用 | 保留 | 八张候选图可从颜色与边缘区分两件物品 |
| 浅色布 → 橙色清洁布 | A 被挂回；B 有独立 association 并用于清理灶台积水 | 保留 | V1 中橙色目标稳定、区分度高 |
| 刀具 A → 刀具 B | 旁白说“换一把刀”，但官方 object movement 把多个轨迹归到同一 association | 拒绝 | 不能确定是两个物理实体，不能据此构造 A/B 真值 |
| 另一组刀具 A → 刀具 B | 两把刀有不同 association，A 放回后 B 被拿起并用于切鸡肉 | 拒绝 | 4+4 拟发布图中 A 大部分被刀架遮挡，两把可见刀柄都接近黑色，无法只凭画面映射到两个不同网页商品；不能用“另一把/尺寸合适”的旁白泄漏替代视觉证据 |
| 毛巾 A → 毛巾 B | 后段同时持有并使用两条毛巾 | 拒绝 | 最终状态不是单一 B，单商品 checker 会强加视频中不存在的真值 |

EgoSchema/Ego4D 的 retain 候选不要求证明“A 被放回、B 成为最终选择”，因为它测量的是后续视觉干扰到来时是否仍遵循明确的“第一件”指令。当前 3 条分别为：深灰黑格纹衬衫后出现蓝色格纹衬衫、浅色拼色毛衣后出现藕紫色罗纹毛衣，以及藏蓝白格纹衬衫后出现红黑格纹衬衫。两条衬衫候选来自同一上游 Ego4D 长视频，不能作为两个独立统计样本直接重复计数。蓝色坚果陈列片段经 4 fps 与 8 fps 密集复核仍只显示伸手，没有独立商品被取出，已按 `ground_truth_not_deterministic` 淘汰。

对应的单一事实源是：

- 连续事件、原始标注、object association、4+4 时间戳和拒绝理由：`data/sources/refresh_external_screening_manifest.json`
- 全量关键词扫描的固定正则、源文件哈希和 87 条命中：`artifacts/reports/mining/hdepic_refresh_keyword_hits.json`
- 每张截图的原视频时间、片段内时间和 SHA-256：`data/sources/refresh_external_frame_provenance.json`
- EgoSchema retain 事件、3 条拒绝记录和 4+4 时间戳：`data/sources/refresh_retain_screening_manifest.json`
- EgoSchema retain 截图的源时间和 SHA-256：`data/sources/refresh_retain_frame_provenance.json`
- 7 个 EgoSchema 片段的下载哈希、源 Ego4D 区间和筛选状态：`artifacts/reports/mining/refresh_external_candidates.json`
- 实际任务截图：`data/observations/`；保留的八图审核材料：`artifacts/reports/source-evidence/`

这些结果仍属于构造工作台，不是 Benchmark Release。`video_screened` 只表示“连续源事件与八张候选输入没有明显冲突”；它还不代表商品目录、任务 checker、盲审或真实 Agent 运行已经通过。尤其是餐巾和零食袋事件必须先通过图像盲审，若两名评审不能稳定确认关键事实，就应直接淘汰。

## 参考链接

### 主数据源

- HD-EPIC: https://hd-epic.github.io/
- HD-EPIC paper: https://arxiv.org/abs/2502.04144
- HD-EPIC annotations repo: https://github.com/hd-epic/hd-epic-annotations
- HD-EPIC data page: https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47
- EPIC-KITCHENS: https://epic-kitchens.github.io/
- EPIC-KITCHENS-100: https://epic-kitchens.github.io/2020-100
- EPIC-KITCHENS-100 annotations repo: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- VISOR: https://epic-kitchens.github.io/VISOR/
- Ego4D: https://ego4d-data.org/
- Ego4D download docs: https://ego4d-data.org/docs/start-here/
- Ego4D Hands & Objects docs: https://ego4d-data.org/docs/benchmarks/hands-and-objects/
- Ego-Exo4D: https://ego-exo4d-data.org/
- Ego-Exo4D download docs: https://docs.ego-exo4d-data.org/download/
- AEA: https://www.projectaria.com/datasets/aea
- AEA docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset
- AEA download docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_download_dataset

### 邻接 / 高召回数据源

- EGTEA Gaze+: https://cbs.ic.gatech.edu/fpv/
- ADL dataset page: https://web.cs.ucdavis.edu/~hpirsiav/papers/ADLdataset/
- ADL paper PDF: https://www.cs.cmu.edu/~deva/papers/ADL_2012.pdf
- Charades-Ego: https://prior.allenai.org/projects/charades-ego
- Charades-Ego paper: https://arxiv.org/abs/1804.09626
- EgoLife repo: https://github.com/evolvinglmms-lab/egolife
- EgoLife dataset card: https://huggingface.co/datasets/lmms-lab/EgoLife
- EgoLife paper: https://arxiv.org/abs/2503.03803
- UT Ego: https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html

### 派生层

- EPIC-SOUNDS: https://epic-kitchens.github.io/epic-sounds/
- EPIC-SOUNDS annotations repo: https://github.com/epic-kitchens/epic-sounds-annotations
