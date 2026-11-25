# 审核材料与历史实验

本目录不提交到 Git，但忽略提交不等于可以删除。新模型执行实验统一写入项目根目录`experiments/`；本目录保留已有实验与审核依据。单次运行用 `uv run egocua inspect-run <运行目录>` 查看。

- `runs/`：历史真实运行档案，包括任务与网站快照、动作和结束状态。必须保留；重新运行不能替代原记录。
- `human-review-packages/`：固定40题质量核查的浏览器离线包及ZIP。只包含登记材料、预设答案和页面，不包含既有人工意见、模型成绩或密钥；审核者解压后直接打开HTML，完成后导出JSON回传。
- `human-answer-validation/`：独立参与者的答案选择、草稿、提问和首次提交。样本由原40题登记引用；不属于模型执行成绩，不能当作任务状态。
- `reports/`：检查报告、统计和审核材料。统计可从原始记录重算；Coding Agent 的具体评语、审核历史及其引用的材料必须保留。
- `studies/`：历史实验的计划、结果和分析。保留已完成实验的结果及运行引用。
- `screening/`、`closure-packets/`：筛选与审核展示材料。清理前检查任务和审核记录是否引用。
- `cache/`：可重新生成的检查缓存。
- `superseded/`：历史输出，不代表当前任务状态。

旧运行按原路径归档，避免断开快照和逐条评语引用。切换前的试跑汇总在 `superseded/task-qualification-pre-20-step-20261104.json`；当前试跑汇总只纳入 20 步固定回放。

正式输入画面在 `data/observations/`。任务状态只从 `egocua synthesize status` 读取。
