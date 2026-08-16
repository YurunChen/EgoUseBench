# 首轮 Ego2Web 种子筛选报告

> 2026-08-15 当前状态：本报告是历史筛选记录，不是当前任务清单。手指玩偶 Clarify 已因可被单张照片等价替代而退出核心集合；当前三道种子以 [`data/selection.json`](../../data/selection.json) 为准。
>
> 2026-08-09 后续状态：本报告保留首轮 **Ego2Web 内部筛选** 的原始结论。外部数据源审计随后从 HD-EPIC 筛出 4 条 `video_screened` Refresh-replace 候选，并淘汰 3 条真值不唯一或发布图像无法区分目标的事件；详见 [`refresh-source-dataset-audit.md`](refresh-source-dataset-audit.md) 和外部筛选清单。下面“Refresh 缺种”仅描述本轮 Ego2Web 池，不再代表整个项目当前状态。
>
> 同日的缺失 subtype 复核进一步筛出 Graft-C、Graft-R、Clarify-sequence 和 Clarify-position 各 1 条 `video_screened` 候选。#3 实验室容器在 sequence 复核中因候选多于两个再次淘汰，#19/#20 零食货架也因触碰锚点被遮挡且相邻商品过多而淘汰；position 改用 #59 能量饮料货架中的绿色 Monster 左右两个产品角色。详见 [`subtype_expansion_screening_manifest.json`](../../data/sources/subtype_expansion_screening_manifest.json)。

## 结论

首轮连续视频复核曾保留 Graft-P、Refresh-replace 和 Clarify-variant 三条待裁决种子。随后按 Agent 真正能够检查的图片重新审查并完成修正：Graft 保留 85–105 秒的 8 张 UNO 输入；Clarify 改为 0–35 秒均匀 8 张，使前半段主要操作猪、后半段主要操作兔子；Refresh 经过 9–21 秒的 4 fps 密集复核后淘汰。当前机器清单是 2 条 `video_screened` 与 12 条 `rejected`，可重复抽取的当前输入共 16 张。

Refresh-replace 的结果最值得注意：9 个最可能包含目标变化的视频全部不符合。最后一条零食棒序列只显示手依次接触红色 LUNA、下层橙色商品和黄色 LUNA 区域，没有证明“选择 A、放回或放弃 A、最终保留唯一 B”；原先拟发布的 V1 也跨越多个候选区域。因此该种子已经退出审核队列，不写任务、不构造目录，也不改成 retain 填补配额。

当前保留的 Graft 与 Clarify 仍只到 `video_screened`，不能称为已生成任务或发布级种子。它们还没有通过双人构造复核、商品目录构造、独立人类完成和真实模型闭环；Refresh 轨目前没有合格种子。

这轮筛选最重要的结果不是“凑齐三条”，而是确认自动按文本限定词或时间段配对会制造错误构念。Clarify 必须在原视频中真的同时存在两个合理候选；Refresh 的 V1 必须在时间上晚于 V0，并且确实改变或保持用户所指；Graft 的后续要求必须自然且必要，不能把网页标注中的第二种商品直接当成视频里发生的用户更新。

权威机器可读记录是 [`seed_screening_manifest.json`](../../data/sources/seed_screening_manifest.json)。本报告只解释筛选判断，不复制任务真值。

## 等待人工裁决的两条候选

| 候选 | 视频证据 | 当前能支持的判断 | 尚不能支持的判断 | 下一道门禁 |
|---|---|---|---|---|
| `seed-graft-p-01` | 85–105 秒均匀 8 帧，8 张都显示同一局 UNO | 输入能稳定识别 UNO；可研究保留标准版并加入旅行/铁盒版 | 旅行/铁盒版未作为实物出现；B 必须缩成不混入零售商和汇报差异的最小反事实补充 | 两名构造者判断最小 B 是否自然且必要；通过后再链接网页资产 |
| `seed-clarify-variant-01` | 0–35 秒均匀 8 帧；8 张都同时显示白兔和粉猪，前 4 张偏向操作猪、后 4 张偏向操作兔 | 恰有两个同类候选；动作显著性不再只偏向隐藏目标；一个角色属性可唯一消歧 | 原 Ego2Web 标注要求两个角色都要；二选一是新构造的反事实任务，不是删去限定词 | 两名构造者验证回答前无稳定默认目标、一次回答唯一性和新请求的自然性；通过后再构造网页资产 |

### 实际输入数量与频率

| 候选/packet | 图像数 | 连续区间 | 相邻图像间隔 | 等效频率 |
|---|---:|---:|---:|---:|
| Graft `V0` | 8 | 85–105 s | 2.857 s | 0.350 Hz |
| Clarify `V0` | 8 | 0–35 s | 5.000 s | 0.200 Hz |

频率不是手工设置的固定 fps，而是由“连续区间长度 ÷ 相邻帧数”派生。当前两条候选都维持 8 张上下文预算；Clarify 扩大时间区间不是为了补充更多图片，而是让同样的 8 张覆盖两个候选各自被操作的过程。首轮只把这一方案当作受控 pilot 协议，不声称 8 张是最优值；后续如做帧数消融，应另设 4/8/16 张的实验计划，不能在主结果中为不同任务临时加帧。

### 逐任务设计结果

#### `seed-graft-p-01`

这条任务保留，诊断的是“用户后来新增一个并行目标时，Agent 能否保留原目标并完成两者”。具体任务语言已经固定为：

- A：找到正在玩的卡牌游戏的标准版，加入购物车并提交；
- B：再加入同款游戏的便携/铁盒版，保留标准版，最后一起提交；
- bundled：开始时一次性要求标准版和便携/铁盒版各一件。

最终 checker 只接受“标准版一件 + 便携/铁盒版一件、无额外商品”。这样 `bundled` 和 `injected` 的最终任务完全相同，只改变 B 的到达时间；`a_only` 只检查标准版。Amazon、其他零售商、价格汇报和包装说明不进入 B，避免同时改变多个因素。视频只负责确定游戏是 UNO，旅行版来自原 Ego2Web 标注并明确记为 `counterfactual_recombination`，不伪装成视频中真实出现的第二副牌或口头打断。

#### `seed-refresh-replace-01`

这条候选已经以 `event_not_visible` 淘汰。9–21 秒的密集序列只能显示购物者接触多个商品区域，无法确定唯一 B，连续证据也没有证明 A 被放弃后最终保留 B。因此：

- 不写正式用户指令；
- 不构造商品目录；
- 不把“第二个接触的商品”当作最终选择；
- 不改成 retain 来填补配额；
- 从双人审核队列和当前 provenance 中移除；旧 contact sheet 只作为淘汰审计证据保留。

能替代它的新任务必须来自已有数据中的完整连续事件：用户先选择 A，明确放回或放弃 A，再选择并保留唯一 B；Agent 实际收到的后 4 张也必须独立识别 B。

#### `seed-clarify-variant-01`

这条任务条件性保留，诊断的是“同一画面有兔子和猪两个合理候选时，Agent 是否会在承诺具体商品前询问是哪一个”。具体任务语言已经固定为：

- ambiguous：找到工作台上那个毛毡指偶对应的单角色 PDF 缝纫图样并提交；
- oracle：直接说明要工作台上的粉色猪指偶单角色 PDF 图样；
- 合法问题：`Do you mean the rabbit puppet or the pig puppet?`；
- 标准回答：`I mean the pig puppet.`；
- 最终 checker：只接受一个猪指偶的单角色可下载 PDF 图样，不接受同时包含兔子和猪的套装。

`interactive` 与 `forced_execute` 使用完全相同的歧义输入，只相差一次提问工具；`oracle` 用于测量不必要提问。原来的 0–15 秒输入持续操作猪，与“我正在制作的”组合后会泄漏默认目标；新版改为 0–35 秒并删除这句话，使猪和兔子的动作证据对称。原 Ego2Web 标注要求两个角色一起出现，因此这仍不是“删除限定词”，而是基于真实双候选画面构造的新二选一意图。两名构造者仍须独立确认回答前没有稳定默认目标，并确认一次回答即可唯一确定目标。

### Graft-P 视频帧

![Graft-P UNO contact sheet](../../artifacts/reports/source-evidence/e7df84b5586f8f3c9974815486e4e0284338fd7852b435d77d759a84e351cca5.jpg)

### Refresh-replace 视频帧

该 Refresh 种子在后续构造审计中被淘汰，因此不再保留为当前任务审核图。

### Clarify-variant 视频帧

![Clarify finger-puppet contact sheet](../../artifacts/reports/source-evidence/1047c813f455a6565b0b9a1b7452ad10331311f85460b2975f02087a5eadbd3b.jpg)

## 淘汰记录

| 来源候选 | 主淘汰类型 | 淘汰原因 | 可否改作其他类型 |
|---|---|---|---|
| 零食棒候选 `seed-refresh-replace-01`，#144/#23 | `event_not_visible` | 9–21 秒密集复核只显示接触多个区域，没有唯一的 A 放回后最终保留 B；拟发布 V1 也无法唯一识别 B | 否；不能靠序数标注或改成 retain 复活 |
| 原 `clarify-01`，Ego2Web #1 | `insufficient_candidate_ambiguity` | 只有一个控制器，没有两个视频支持的控制器变体 | 否；不能靠网页目录补造现实歧义 |
| 原 `clarify-02`，Ego2Web #3 | `construct_not_isolated` | 多个玻璃容器的自然差异是先后顺序，不是商品变体 | 可重新进入 sequence 候选池，但必须重新构造，不能直接改标签通过 |
| 原 `clarify-03`，Ego2Web #7 | `insufficient_candidate_ambiguity` | 只有一件相关运动上衣，没有两个合理候选 | 否 |
| 食品连续拿取，#4/#5/#6 | `construct_not_isolated` | 购物者依次收集多个独立清单商品，没有放弃前一个目标 | 可用于 Graft 多目标池，不能作为 Refresh-replace |
| 实验室玻璃容器，#3 | `construct_not_isolated` | 多个容器属于实验流程的先后步骤，不是消费者换选 | 可重新评估 Clarify-sequence，不能作为 Refresh-replace |
| 白色外套，#0/#142 | `event_not_visible` | 白色外套被放回，但之后只是继续逛店，没有最终选择另一件衣物 | 否；缺少 B 的最终选择 |
| 能量饮料，#58/#59 | `temporal_relation_not_supported` | 一个商品被选中，另一个只是相邻比较对象 | 可用于 Clarify 或比较任务，不能作为 Refresh-replace |
| 糖果，#191/#192 | `event_not_visible` | 相关标注始终指向同一个商品，后续没有完整 A→B 替换 | 否 |
| 方便面，#228/#230 | `behavior_change_not_required` | 两条标注都指向同一次选择，没有第二目标 | 否 |
| 酱料/零食，#31/#352 | `temporal_relation_not_supported` | 自动提议的 V1 时间段早于 V0，而且两个片段属于不同类别 | 当前配对否 |
| 相邻零食，#19/#20 | `event_not_visible` | 只看见触碰或指向相邻商品，没有完整选择、放回和最终选择 | 可考虑 Clarify-position，不能作为 Refresh-replace |

## 当前 Gate 状态

| Gate | 状态 | 说明 |
|---|---|---|
| 连续视频与实际输入证据 | 部分通过 | Graft/Clarify 的 16 张当前输入已有均匀时间戳、路径、SHA-256 和派生频率；Refresh 因连续事件和 V1 唯一性不成立而淘汰 |
| 构念隔离 | 部分通过 | Graft 的 B 自然性与 Clarify 回答前无默认目标仍须双人判断；Refresh 当前无合格样本 |
| 冻结网页目录 | 未通过 | 尚未构造带来源、许可说明和文件哈希的 16 商品目录 |
| 双人构造一致性 | 未通过 | 不能由当前代码运行代替人工判断 |
| 独立人类可完成性 | 未通过 | 尚未在精确 harness 下完成三种实验臂 |
| 真实模型闭环 | 未通过 | 只有门禁代码和测试夹具，不存在可计入科研结果的 GPT-5.4-mini 运行 |
| 发布资格 | **No-Go** | 只有两条筛选候选且 Refresh 缺种；当前没有可发布 Benchmark Release |

## 下一步

先完成 Graft 与 Clarify 的双人构造判断，同时从已有第一人称数据集中重新挖掘自然“放回 A 后最终拿走 B”的 Refresh 连续片段。三轨各有一条通过视频与输入门禁的种子后，才为通过者构造冻结目录和正式 task JSON，再进行独立人类运行、evaluator replay 和 Browser-Use + GPT-5.4-mini 的协议有效性验证。模型结果只用于验证运行链路，不能用于决定保留哪条任务。

新 Refresh 必须让完整视频和 Agent 实际收到的后 4 张共同支持唯一最终 B；不能把“依次接触多个物体”降格解释为替换，也不能用模型生成画面补足缺失动作。
