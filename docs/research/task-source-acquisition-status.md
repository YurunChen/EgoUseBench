# EgoCUA-Bench 三轨数据源与实际采集状态

_Last checked: 2026-08-10._

这份文档回答两个容易混在一起的问题：哪些第一人称数据集在概念上相关，以及哪些数据已经实际下载、逐段检查并生成了 Agent 会看到的截图。只有后者才可能进入任务构造；“论文中有相似场景”不能算任务数据。

## 结论

- **60 条主任务的来源筛选已按预定配额冻结。** 当前另留 1 条 reserve；这 61 条候选都已有连续视频依据、任务蓝图和 8 张拟发布输入，但仍只是 `video_screened`，不能称为可发布任务。
- **Graft 的 30 条任务仍全部来自 Ego2Web。** 原因不是它的视频质量最高，而是它同时给第一人称视频和可连接网页执行的成对任务标注，适合构造 C/P/R 三类后续指令变化。
- **Refresh 同时使用 HD-EPIC、Ego2Web 和 EgoSchema/Ego4D。** HD-EPIC 的对象 association 主要支持 replace；Ego2Web 与 EgoSchema/Ego4D 的连续服装和物体片段补充 retain。来源文本只用于定位，最终目标必须由拟发布图像和连续视频共同支持。
- **Clarify 以 Ego2Web 的连续双候选画面为主，并加入两条 HD-EPIC 候选。** variant、sequence、position 均达到冻结配额；EgoPoint-Bench 与 REI-Bench 仍只支撑动机和审核规则，不直接充当任务真值。
- **当前完成的是输入覆盖和候选任务编译，不是任务验收。** 60 条 primary 已连接到同一个固定 EgoMart 站点并可进入 deterministic smoke，另有 1 条 reserve。此前基于任务级 16 商品短名单得到的预审结论不能证明固定站点中的搜索结果无碰撞；所有候选仍缺两名隔离审核者的结论。

## 数据源如何按轨道分工

| 数据源 | Graft | Refresh | Clarify | 当前决定 |
| --- | --- | --- | --- | --- |
| [Ego2Web](https://github.com/Yui010206/Ego2Web) | 强：同视频多条 Web 指令可构造 A/B | 中：补充 retain，replace 证据较弱 | 中强：可从连续画面找真实双候选 | Graft/Clarify 主源，并补充 Refresh-retain |
| [HD-EPIC](https://hd-epic.github.io/) | 弱：没有现成 Web 指令对 | 很强：对象 association 与 pickup-to-placement 区间 | 中：多对象场景多，但没有现成歧义用户意图 | 当前 Refresh 主源 |
| [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2020-100) + [VISOR](https://epic-kitchens.github.io/VISOR/) | 弱 | 强：动作段加手-物 mask，适合下一轮扩展 | 中 | 标注仓库已固定，媒体待 pilot 稳定后下载 |
| [Ego4D](https://ego4d-data.org/) / [EgoSchema](https://github.com/egoschema/EgoSchema) | 中：可构造反事实后续要求，但缺原生 Web B | 强于 retain：服装片段可形成“保持最初对象”场景；replace 仍缺对象级监督 | 中：多候选长场景，但缺 exactly-two 真值 | 当前保留 3 个 retain 候选，其中 1 个因同源 cluster 只作 reserve |
| [EgoPoint-Bench](https://github.com/GUYYYUG/EgoPoint) | 不适合 | 不适合 | 支撑指代/邻近干扰动机，但单张指向通常已给出目标 | 不直接转为 Clarify 任务；避免把 grounding 错写成 ambiguity |
| [REI-Bench](https://github.com/jcx0110/REI-Bench) | 不适合 | 不适合 | 支撑“模糊指令需要澄清/规划”的构念 | 仅作邻接基准；模拟器画面不混入真实视频任务 |
| Ego-Exo4D / AEA / EgoLife / UT Ego | 中到弱 | 适合 retain 或长历史补充 | 中到弱 | 记录为后续池，不在本轮批量下载 |

## 已实际保存的数据

### HD-EPIC

- 本地源片段：`reference/datasets/HD-EPIC/clips-screened/`
- 官方标注快照：`reference/HD-EPIC-annotations/`
- 全量 narration 固定正则扫描：87 行 / 33 个视频，结果位于 `artifacts/reports/mining/hdepic_refresh_keyword_hits.json`；命中只用于排队回看。
- 多轮筛选后当前保留 11 个候选：8 个 Refresh-replace、1 个 Refresh-retain、2 个 Clarify-variant。两个已构造的 replace 候选分别因 V0 已显示后续纸巾目标、指令与 V0 已泄漏后续海绵目标而被淘汰；新增的烤盘切换片段补回 replace 配额。
- 当前 11 个候选对应 88 张输入；两条被淘汰任务的 16 张仍保留作审计证据，因此 HD-EPIC provenance 共校验 104 张。
- 八图总览：`artifacts/reports/source-evidence/`
- 原视频时间、片段内时间和图片哈希：`data/sources/refresh_external_frame_provenance.json`

### EgoSchema / Ego4D

- 7 个官方三分钟片段：`reference/datasets/egoschema/videos/`
- 3 个 retain 候选各生成 8 张输入，共 24 张：`data/observations/`
- 八图总览：`artifacts/reports/source-evidence/`
- 连续事件、任务蓝图与 3 条拒绝记录：`data/sources/refresh_retain_screening_manifest.json`
- 每张截图的源时间与 SHA-256：`data/sources/refresh_retain_frame_provenance.json`
- 粗筛索引与仍待核查的片段：`artifacts/reports/source-evidence/` 和 `artifacts/reports/mining/refresh_external_candidates.json`
- 两条格纹衬衫候选来自同一条上游 Ego4D 长视频的不同区间；small release 最多接纳一条，否则统计必须按 source cluster 聚类。

### Ego2Web

- 本地数据包含 500 条任务标注和 310 段视频；全量结构检索结果保存在 `artifacts/reports/mining/divergence_mining.json`。
- 首批 Graft-P 和 Clarify-variant 共 16 张输入：`data/observations/`
- 新增 Graft-C、Graft-R、Clarify-sequence、Clarify-position 共 32 张输入：`data/observations/`
- 新增任务分别使用木块叠搭游戏的网站评分约束、割草机线盘头与兼容线轴、先后出现的两张海报，以及绿色 Monster 左右两侧的 Rockstar/NOS。
- 多轮筛选清单累计保留 47 条，覆盖 30 条 Graft、12 条 Clarify 和 5 条 Refresh-retain；对应输入共 376 张。

因此，项目当前一共保存并校验了 **504 张来源抽帧**：Ego2Web 376 张、HD-EPIC 104 张、EgoSchema/Ego4D 24 张。其中当前 61 条 screened 候选实际对应 488 张；HD-EPIC 另外 16 张属于后续淘汰纸巾与海绵任务的审计证据，不再作为候选 Agent 输入。各种 dense/overview/contact sheet 只用于构造审核，不计入 Agent 输入。

## 为什么没有直接把更多公开图像变成任务

当前门禁要求来源事件、发布截图、用户语言和网页 checker 指向同一个可验证对象。以下情况即使图片看起来合适，也不能进入任务：

1. 只有单张商品图，无法满足 Graft/Clarify 的 8 张连续上下文预算；
2. 画面中的指向手势已经唯一选中目标，却被改写成“视觉证据不足”；
3. 旁白说“换了一个”，但官方物体关联不能证明是两个实体；
4. A 与 B 都被最终保留，却强行设置单一 B checker；
5. 只有对象图片，没有自然且必要的 Web 指令变化；
6. 只能从数据集答案或标注文本知道目标，Agent 实际收到的图片看不出来。

这也是为什么本轮没有把全量 500 条 Ego2Web 标注直接改成任务：当前 32 条记录已因来源事件、输入证据或最终目标不成立而明确淘汰。除早期来源筛查失败外，两条已构造任务说明淘汰也可能发生在目录和模型诊断阶段：纸巾卷在 V0 已经明显可见；海绵任务的指令直接把后续对象限定为 sponge，而 V0 是金属清洁球，三次 stale 运行也稳定选中最终海绵。两者都不需要 V1 才能确定 B，因而不再测视觉更新。HD-EPIC 的第二个换刀片段虽有两个官方对象 association，也因八张拟发布图看不出稳定商品差异而被淘汰；EgoSchema 两个片段则因行为变化不成立或发布图像不能稳定展示对象而被淘汰。

## 下一步数据工作

1. 先对三轨代表性种子完成双人隔离审核；审核同时绑定源输入、任务、固定站点商品库与任务引用商品审计图的 SHA-256。
2. 对 60 条 primary 逐条检查在固定站点中的真实搜索结果与同名/同图碰撞；未通过者修改站点文本或移出本轮，而不是恢复任务级可见商品过滤。
3. 对共享同一上游长视频的候选执行 source-cluster 限制，避免把相邻生活过程当成独立数据点。
4. 三轨至少各有一条通过双人构造审核、网页人工完成、evaluator replay 和真实模型运行后，再批量扩展其余目录。
5. 当前不要求扩展到 60 条可执行任务；先冻结通过全部门禁的当前子集，再根据科研问题覆盖与统计功效决定是否补样。
