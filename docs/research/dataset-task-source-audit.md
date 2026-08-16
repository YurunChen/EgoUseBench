# EgoCUA-Bench 任务来源数据集审计

更新时间：2026-08-12

## 结论先说

当前最值得继续挖、并且已经能直接服务任务生产的来源，只有三层优先级：

1. **第一优先：Ego2Web**  
   这是目前唯一同时满足“第一人称连续视频 + 已经证明能连到网页任务”的公开来源。它最适合继续扩展 **Graft**，也最适合补 **Clarify**，还能继续挖一部分 **内容筛选** 和 **地图路线** 任务。

2. **第二优先：HD-EPIC + EPIC-KITCHENS-100/VISOR**  
   这组数据最适合补 **Refresh**。原因不是它们更像网页任务，而是它们对“拿起 A、放下 A、换成 B”“手和物体接触”“物体轨迹和遮挡”这些变化标得最清楚，最容易把目标更新或目标保持说清楚。

3. **第三优先：Ego4D / EgoSchema**  
   这组数据适合补长时序的 **Refresh-retain**，也就是用户最开始看过什么、后面画面变了但目标不该变。它们覆盖场景广，但原生并不是网页任务，需要后续再把视频证据映射到网页站点。

其他数据集大多只能当补充或参考。它们常见的问题有四类：

- 只有动作识别，没有自然网页任务连接；
- 只有对象框或短片段，不足以表达中途变化；
- 场景过于专门，比如工业装配、体育动作、手势；
- 虽然有丰富多模态，但公开可下载范围、许可或隐私边界还不适合作为当前主力来源。

## 我们判定“可用于任务生成”的标准

一个数据源至少要尽量满足下面几条：

- **有真实可下载的视频**，不是只有特征或只读排行榜；
- **视频是连续的**，能看到“先看到什么、后来又发生了什么”；
- **有一种以上可核查线索**，例如对象、动作、位置、叙述、音频、时间段；
- **能自然连到网页操作**，至少能连到商品选择、教程/内容筛选、地点/路线三类中的一类；
- **能支持最终状态检查**，而不是只能问一个开放问题。

## 总表

| 数据集 | 一手来源 | 公开内容与标注 | 访问与许可 | 适合哪类轨道 | 适合哪类网页任务 | 结论 |
|---|---|---|---|---|---|---|
| **Ego2Web** | [项目页](https://ego2web.github.io/), [论文](https://arxiv.org/abs/2603.22529), [代码](https://github.com/Yui010206/Ego2Web), [数据集](https://huggingface.co/datasets/Shoubin/Ego2Web) | 官方定义就是“从第一人称视频到网页执行”。覆盖 E-Commerce、Media Retrieval、Knowledge Lookup、Local/Maps。项目页明确说任务来自真实第一人称视频并要求在线执行。 | 数据已公开；代码和数据页可访问。Hugging Face 数据页可预览样例。项目页未清楚说明上游视频再分发边界。 | **Graft 最强**；Clarify 强；Refresh 中等 | **商品强，内容强，地图强** | **立即可用，当前主源** |
| **Ego4D** | [项目页](https://ego4d-data.org/), [Start Here](https://ego4d-data.org/docs/start-here/), [代码](https://github.com/facebookresearch/Ego4d) | 官方主页给出 3,670 小时、923 位参与者、74 个地点、9 个国家。含大规模连续第一人称视频与多种 benchmark。Episodic Memory 文档给出 NLQ/VQ/MQ 共约 74K queries，覆盖约 800 小时视频。 | 需签署许可，官方说明通常约 48 小时后发放下载凭证。 | **Refresh 强**；Clarify 中；Graft 中 | 商品中，内容中，地图中 | **需申请/下载，后续核心来源** |
| **EgoSchema** | [项目页](https://egoschema.github.io/), [论文](https://arxiv.org/abs/2308.09126), [代码](https://github.com/egoschema/EgoSchema) | 来自 Ego4D 的三分钟长片段。项目页写明有 5K+ 人工多选题，覆盖 250+ 小时真实视频。提供下载脚本和 Google Drive 视频。 | 可直接下载脚本或 Kaggle/Drive；实际视频仍承接 Ego4D 上游边界。 | **Refresh-retain 强**；Clarify 中；Graft 弱 | 商品中，内容弱到中，地图弱 | **立即可用于补长时序，但不是主源** |
| **HD-EPIC** | [项目页](https://hd-epic.github.io/), [论文](https://arxiv.org/abs/2502.04144), [数据页](https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47), [标注代码](https://github.com/hd-epic/hd-epic-annotations), [下载器](https://github.com/hd-epic/hd-epic-downloader) | 官方项目页强调对象、动作、音频、手部 mask、gaze、3D 数字孪生。标注仓库明确提供 object movement associations：从 pick-up 到 placement 的时间段、bbox、3D 位置、scene fixture。 | 数据页显示整包 2.3 TiB；CC BY 4.0；下载器支持按 `--videos`、`--audio`、`--hands`、`--digital-twin` 等子集下载。 | **Refresh 最强**；Clarify 中；Graft 弱 | 商品中，内容中，地图弱 | **立即可用，当前 Refresh 主源** |
| **EPIC-KITCHENS-100** | [项目页](https://epic-kitchens.github.io/2021), [标注仓库](https://github.com/epic-kitchens/epic-kitchens-100-annotations) | 官方项目页写明 45 个厨房、100 小时、20M 帧、90K 动作段、20K narrations、97 verbs、300 nouns。动作、叙述、自动手/物体标注都可获取。 | 研究用途公开；视频可从 Data.Bris 或学术种子下载；版权为 CC BY-NC 4.0。 | **Refresh 强**；Clarify 中；Graft 弱 | 商品中，内容中，地图不适合 | **需下载，适合扩充厨房物体任务** |
| **VISOR** | [项目页](https://epic-kitchens.github.io/VISOR/site) | 官方页写明 271K 稀疏 masks、14.9M dense masks、67K in-hand object masks，并可交互浏览长序列。它直接补足手-物接触和遮挡。 | 注释和稀疏帧可下载；CC BY-NC 4.0；依赖 EPIC-KITCHENS 视频。 | **Refresh 强**；Clarify 中强；Graft 弱 | 商品中，内容中，地图不适合 | **需与 EPIC-KITCHENS 联合使用，优先级高** |
| **Ego-Exo4D** | [项目页](https://ego-exo4d-data.org/), [论文](https://arxiv.org/abs/2311.18259), [Project Aria 文档](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/ego-exo4d) | 官方论文介绍为同步 ego/exo 的 skilled activity 数据，含多通道音频、眼动、点云、姿态等。官网写明 740 位参与者，数据于 2023-12-13 公开。 | 需单独签署 Ego-Exo4D 许可；即使已有 Ego4D 许可也要重新签。 | Refresh 中；Clarify 中；Graft 弱 | 内容中，商品弱，地图弱 | **需申请，适合教程/技能类补充，不是当前主力** |
| **EgoExoLearn** | [代码与数据入口](https://github.com/OpenGVLab/EgoExoLearn), [论文](https://arxiv.org/abs/2403.16182) | 官方 repo 写明数据包含约 120 小时 ego/exo procedural activities，带高质量 gaze 和多模态标注，并直接提供 Google Drive、Baidu、Hugging Face 下载。 | 可直接下载，公开入口明确。 | Refresh 中；Graft 中；Clarify 弱到中 | **内容筛选最合适**；商品弱；地图不适合 | **需下载，可作为教程类第二梯队** |
| **Assembly101** | [项目页](https://assembly-101.github.io/), [下载脚本](https://github.com/assembly-101/assembly101-download-scripts) | 官方页写明 4321 视频、101 个玩具装配/拆卸、8 路静态 + 4 路 ego，同步多视角；有 100K coarse actions、1M fine-grained actions、18M 3D hand poses，还有 mistake detection。 | 2026 年 5 月迁移到 Hugging Face，公开更方便；CC BY-NC 4.0。 | Refresh 中；Graft 中；Clarify 弱 | **内容筛选中**；商品弱；地图不适合 | **需下载，可补“找教程/改教程条件”类，但场景偏专门** |
| **MECCANO** | [项目页](https://iplab.dmi.unict.it/legacy/MECCANO/), [官方仓库](https://github.com/fpv-iplab/MECCANO) | 官方 repo 写明这是工业场景的多模态第一人称数据，含 RGB、depth、gaze；支持 action recognition、active object、EHOI、anticipation、next-active-object。 | 可从官网下数据和标注；公开可获取。 | Refresh 中；Clarify 弱到中；Graft 弱 | 内容弱到中；商品弱；地图不适合 | **仅参考或少量补充，域太窄** |
| **ADL** | [官方页面](https://web.cs.ucdavis.edu/~hpirsiav/papers/ADLdataset/), [论文 PDF](https://www.cs.cmu.edu/~deva/papers/ADL_2012.pdf) | 官方页写明有 1M 帧、非脚本化日常活动，标注 activities、object tracks、hand positions、interaction events，并直接提供 43.1 GB 视频与标注下载。 | 可直接下载；老数据集，规模和分辨率都偏旧。 | Refresh 中；Clarify 弱到中；Graft 弱 | 商品弱到中；内容弱；地图不适合 | **可用但不是优先级** |
| **EgoTracks** | [Ego4D 文档](https://ego4d-data.org/docs/data/egotracks/), [论文](https://arxiv.org/abs/2301.03213) | 来自 Ego4D 的长期视觉跟踪标注。文档说明可通过 CLI 下载 annotations，结构围绕 visual queries 和目标轨迹。 | 依赖 Ego4D 许可与下载。 | **Refresh 参考价值高**；其他弱 | 不直接对应网页任务 | **仅参考，不单独当任务源** |
| **EgoObjects** | [Meta 数据页](https://ai.meta.com/datasets/egoobjects-dataset/), [代码](https://github.com/facebookresearch/EgoObjects), [下载页](https://ai.meta.com/datasets/egoobjects-downloads/) | Meta 官方数据页写明它是面向第一人称对象理解的大规模对象数据；初始发布 12K+ 视频、40+ 小时、200 个主对象类。GitHub repo 又给出 challenge version：从 9K+ 视频采样 114K 帧，14.4K 唯一实例、368 类别。 | 可下载；更偏对象检测和实例识别。 | Clarify 弱到中；Refresh 弱；Graft 弱 | 商品弱到中；内容不适合；地图不适合 | **仅参考，缺自然连续事件** |
| **Charades-Ego** | [项目页](https://prior.allenai.org/projects/charades-ego), [论文](https://arxiv.org/abs/1804.09626) | 官方页写明有 7,860 个室内活动视频、68,536 个时间标注、157 个动作类，且是 first/third person 成对录制。 | 公开可访问。 | Refresh 弱到中；Clarify 弱；Graft 弱 | 内容弱；商品弱；地图不适合 | **仅参考，场景脚本化且网页连接弱** |
| **EgoLife** | [项目页/博客](https://egolife-ai.github.io/blog/), [代码](https://github.com/evolvinglmms-lab/egolife), [论文](https://arxiv.org/abs/2503.03803) | 官方项目页写明 6 位参与者共同生活 1 周，300 小时级多模态数据，含第一人称视频、音频、眼动、IMU、GPS、15 路第三视角、转录、dense caption、30 秒 long caption，并明确出现 shopping 等生活场景。 | 页面给了数据入口，但当前公开部分较碎；同时涉及明显隐私和人物关系信息。 | Graft 中；Refresh 中；Clarify 中 | 商品中，内容中，地图中 | **仅参考，当前不建议作为主力生产源** |
| **WEAR** | [项目页](https://mariusbock.github.io/wear/), [论文](https://dl.acm.org/doi/10.1145/3699776), [代码](https://github.com/mariusbock/wear) | 官方页写明 22 位参与者、18 类户外运动、同步 egocentric video + inertial data，提供原始视频和加速度数据下载。 | 可直接下载，raw 约 164GB。 | Refresh 弱；Clarify 弱；Graft 弱 | 几乎不适合当前三类网页任务 | **不适合当前任务生产** |
| **EgoGesture** | [论文](https://ieeexplore.ieee.org/document/8299578) | 这是佩戴式手势识别数据，公开描述集中在 RGB-D 手势样本和连续手势检测。 | 公开论文可查，但它的核心就是手势类别，不是自然网页任务。 | 三轨都弱 | 不适合当前三类网页任务 | **不适合当前任务生产** |

## 为什么只有这几类最值得继续挖

### 1. Ego2Web 仍然是唯一的“网页任务原生源”

它的价值不是视频最多，而是它已经把“看现实世界”和“做网页任务”连起来了。对我们现在的三条诊断轨道来说：

- **Graft**：最容易从现有任务里做“先给 A，后给 B”；
- **Clarify**：最容易从现有视频里找出两个合理候选，再把网页目标冻结下来；
- **地图/内容任务**：现成就有 Media Retrieval 和 Local/Maps，不需要从纯动作数据生造网页场景。

本地核对结果：官方发布文件 `reference/Ego2Web/dataset/ego2web.json` 已到盘，统计得到 **500 条任务、310 个视频**；视频目录也已到盘 **310 个视频文件**。

此外，本地已经从 Ego2Web 官方数据里继续抽出了一批后续候选：

- `data/sources/graft_expansion_screening_manifest.json`：27 条 `video_screened`
- `data/sources/clarify_expansion_screening_manifest.json`：9 条 `video_screened`
- `data/sources/maps_seed_screening_manifest.json`：3 条 `video_screened`
- `data/sources/refresh_retain_expansion_screening_manifest.json`：5 条 `video_screened`
- `data/sources/cross_domain_extension_candidate_queue.json`：28 条非电商跨域候选等待继续核查

这说明 Ego2Web 还没有被挖完，当前继续扩任务，最稳妥的路径仍然是先把 Ego2Web 挖深。

### 2. Refresh 的最好来源不是网页 benchmark，而是物体变化标得清楚的数据

**HD-EPIC** 和 **EPIC-KITCHENS/VISOR** 的优势很明确：

- 能清楚看到“先拿哪个、后换成哪个”；
- 有 hand-object 接触和 movement association；
- 有长序列和遮挡恢复；
- 可以把 stale / stream / oracle_latest 这三种条件讲清楚。

对 Refresh 来说，这比“是否原生就是网页任务”更重要。网页部分可以后接我们自己的 mock site，但视觉真值必须先稳。

### 3. Ego4D / EgoSchema 更适合做“保持目标”而不是“直接造网页”

这组数据场景广、时长长，适合做：

- 用户最开始看过的对象，后面离开视野后还要保持；
- 先前目标与后来干扰物并存时，不能被最新画面带偏；
- 长时序下的内容/地点回忆。

但它们的问题也很清楚：

- 原生不是网页任务；
- 很多 benchmark 本身是 QA，不是执行；
- 若直接从问题文本反推任务，容易把 benchmark 的答案泄漏进任务设计。

所以它们适合当 **视觉来源**，不适合直接当 **任务模板来源**。

## 按网页任务类型看，哪些数据最能产任务

### A. 商品选择

最强来源：

- **Ego2Web**：天然有“看见商品/品牌/版本 → 去网页找同款/加购物车”
- **HD-EPIC / EPIC-KITCHENS / VISOR**：天然有“拿起 A、换成 B、相似物并存”
- **EgoObjects**：可当对象实例和干扰项设计参考，但不宜直接当主视频源

不推荐主挖：

- WEAR、EgoGesture、Charades-Ego

### B. 教程与内容筛选

最强来源：

- **Ego2Web**：已有 Media Retrieval
- **Assembly101**：动作步骤、错装、补条件很适合“找对教程/筛掉不满足条件的教程”
- **EgoExoLearn / Ego-Exo4D**：适合“找和当前步骤一致的教程”“补充步骤要求”
- **MECCANO**：可做少量工业教程，但不适合成为主体

### C. 地点与路线

目前真正靠谱的公开来源仍然主要是：

- **Ego2Web**：它已经把 Local / Maps 任务公开成网页执行问题
- **Ego4D / EgoLife**：理论上可能挖到真实外出场景，但网页映射成本高，且当前没有像 Ego2Web 那样现成对齐好的任务层

因此，地图类任务如果要继续扩，**优先仍然应该从 Ego2Web 深挖**，而不是去别的数据集大海捞针。

## 当前建议的继续开采顺序

### 立即继续

1. **Ego2Web 全量继续筛**  
   目标：继续扩 Graft、Clarify、地图、内容筛选。

2. **HD-EPIC 继续筛 Refresh-replace / Clarify**  
   目标：把“换目标”和“双候选歧义”补足。

### 第二批

3. **EPIC-KITCHENS-100 + VISOR**  
   目标：在厨房场景继续扩相似物体、遮挡、换目标、目标保持。

4. **Ego4D / EgoSchema**  
   目标：补长时序 retain，不急着补 Graft。

### 只有在前面不够用时再考虑

5. **Assembly101 / EgoExoLearn / Ego-Exo4D / MECCANO**  
   目标：专门补教程与内容筛选类任务。

## 最终分级

### 立即可用

- [Ego2Web](https://ego2web.github.io/)
- [HD-EPIC](https://hd-epic.github.io/)
- [EgoSchema](https://egoschema.github.io/)（更准确地说，是它能帮我们快速定位长时序 Ego4D 片段）

### 需申请或下载

- [Ego4D](https://ego4d-data.org/)
- [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2021)
- [VISOR](https://epic-kitchens.github.io/VISOR/site)
- [Ego-Exo4D](https://ego-exo4d-data.org/)
- [EgoExoLearn](https://github.com/OpenGVLab/EgoExoLearn)
- [Assembly101](https://assembly-101.github.io/)
- [MECCANO](https://iplab.dmi.unict.it/legacy/MECCANO/)
- [ADL](https://web.cs.ucdavis.edu/~hpirsiav/papers/ADLdataset/)

### 仅参考

- [EgoTracks](https://ego4d-data.org/docs/data/egotracks/)
- [EgoObjects](https://ai.meta.com/datasets/egoobjects-dataset/)
- [Charades-Ego](https://prior.allenai.org/projects/charades-ego)
- [EgoLife](https://egolife-ai.github.io/blog/)

### 不适合当前任务生产

- [WEAR](https://mariusbock.github.io/wear/)
- [EgoGesture](https://ieeexplore.ieee.org/document/8299578)

## 对当前项目最直接的动作建议

如果目标是“尽可能从现有已收集视频里继续生产任务”，最直接的顺序应该是：

1. 先把 **Ego2Web 已到盘 310 个视频** 挖到明显收益递减；
2. 再把 **HD-EPIC 已部分下载的对象变化片段** 扩成 Refresh 主池；
3. 然后决定是否正式申请并下载 **Ego4D / EPIC-KITCHENS / VISOR** 的更大子集；
4. 只有当“教程/内容筛选”数量明显不够时，再进入 **Assembly101 / EgoExoLearn / Ego-Exo4D**。

这样做最省时间，也最符合当前 benchmark 的动机：

- 先保证网页任务真的自然；
- 再补足视觉变化的可验证性；
- 最后才扩更远的数据域。

## 本轮实际落地

本轮按上述顺序继续筛选并排除4道来源与网页任务不一致的题后，当前候选集为 **105 道任务、105 个互不重复的来源视频组、840 张发布画面**：

- Graft 74 道，其中 EgoMedia 任务覆盖规则教程、烹饪、维修、手工、摄影、运动、农作和工具操作；
- Refresh 17 道，其中包含 1 道来自 HD-EPIC 的真实“先用酱油、后来再加入椰奶”变化；
- Clarify 14 道；
- 网页侧为 EgoMart 55 道、EgoMedia 47 道、EgoMaps 3 道。

没有进入任务集的候选继续保留原始拒绝记录，主要原因是关键对象看不清、变化只存在于文字标注、候选不止两个，或与已选任务来自同一来源视频。
