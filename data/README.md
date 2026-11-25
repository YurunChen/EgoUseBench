# 任务材料

正式成员读取 `selection.json::task_set`；可编辑材料位于 `sources/manifests/` 和 `instructions/authored.json`，`tasks/` 是编译结果，不能手改。画面与抽帧依据位于 `observations/`、`sources/provenance/`；研究内容位于 `experiments/`。

网页事实归 `sites/`，新运行归 `experiments/`，历史运行保留在原档案位置。

`reviews/` 保留审核事实；`archive/` 和 `final-tasks/` 保存历史材料与发布快照，不代表当前可用数量。来源的 `source_screened`、`video_screened` 只描述材料进度，任务状态由统一接口计算。

三场景对照从正式原题派生，共用最终答案。Computer-Use 的自包含请求登记在 `experiments/interaction-settings.json`，Offline 使用原题的已审核录像窗口。Cue 描述登记在 `experiments/cue-facts.json`，按原画面变化批次送达。独立实验材料引用原题，不自动加入正式成员。

Git 保留任务定义和必要的来源清单。观察帧、冻结发布快照、原始网页采集和审核记录按原路径保存在本地。媒体的获取条件和许可见[来源登记](../configs/datasets.json)。
