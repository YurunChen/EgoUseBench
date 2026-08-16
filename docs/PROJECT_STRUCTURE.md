# 项目目录

```text
egoCUAbench/
├── egocua/                    # 基准协议和执行代码
│   ├── adapters/              # Browser-Use、人工和模型请求适配器
│   ├── agents/                # 仅用于 EgoMart 调试的最小 Agent
│   ├── schema.py              # Task Schema v2
│   ├── environment.py         # 站点统一入口
│   ├── protocol.py            # 事件和视觉输入时序
│   ├── execution.py           # 一次运行的生命周期
│   ├── final_state.py         # 统一最终状态判断
│   ├── checker.py             # 单次运行评估
│   ├── metrics.py             # 汇总指标和置信区间
│   ├── review_workspace.py    # 本地人工审核的唯一入口
│   └── gates.py               # 审核与发布门禁
├── sites/
│   ├── runtime.py             # 站点共享运行状态
│   ├── content.py             # 文档、媒体、地图共享页面流程
│   ├── egomart/               # 商品与购物车
│   ├── egodocs/               # 保留的资料站实现，当前候选集未使用
│   ├── egomedia/              # 教程、影片与收藏
│   └── egomaps/               # 地点、入口与路线字段
├── data/
│   ├── tasks/{graft,refresh,clarify}/
│   ├── observations/          # 发布给 Agent 的视频抽帧
│   ├── sources/               # 来源证据、时间戳和哈希
│   ├── selection.json         # 当前独立人工审核候选的唯一清单
│   ├── reviews/               # 统一审核规范和两名审核者的独立提交
│   └── releases/              # 仅保存通过全部门禁的版本
├── configs/pilot.json         # 实验计划
├── tools/                     # 数据筛选、审核图和验证工具
├── tests/                     # 接口与端到端测试
├── artifacts/
│   ├── runs/<experiment>/<run-id>/
│   └── reports/
├── docs/                      # 设计、协议和研究记录
└── reference/                 # 只读第三方数据与代码
```

## 单一真相源

1. `data/tasks` 保存任务语义、视觉包路径、站点和最终状态。
2. `sites/<site>` 保存该网站的全部实体内容。任务不复制标题、属性、文章、教程或地点资料。
3. `data/observations` 保存发布帧，`data/sources` 保存它们的来源视频、时间戳和哈希。
4. `configs/pilot.json` 保存每类任务的正式运行方式、重复次数、模型、步数上限和调度种子。
5. `artifacts` 只保存运行与分析结果，不定义正确答案。
6. `data/reviews/construction.json` 保存唯一审核规范，审核范围仍由 `selection.json` 决定。
7. `data/releases` 在任务通过独立审核、人类执行、评测器复放和真实模型运行前保持为空。

## 运行档案

```text
artifacts/runs/<experiment-id>/<run-id>/
├── manifest.json
├── runtime.jsonl
├── events.jsonl
├── trace.json
├── run.json
└── error.json               # 仅失败时存在
```

调试信息跟随具体运行或放在 `artifacts/reports`。不建立第二套任务目录、临时 catalog 或全局 debug 树。
