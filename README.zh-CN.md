<h1 align="center"><img src="website/static/egousebench-logo.png" width="46" alt="">&nbsp; EgoUseBench</h1>

<p align="center"><strong>用户中途改口，电脑操作智能体能跟上吗？</strong></p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong> ·
  <a href="#快速开始">快速开始</a> · <a href="#基准概览">基准概览</a> · <a href="LICENSE">MIT</a>
</p>

**EgoUseBench** 评测智能体如何根据第一人称画面操作网页。智能体在找商品、收藏教程或标记地点时，用户可能更新要求，画面也可能带来新证据。评分只看**最终网页状态**是否符合最终任务要求。

## 41 秒看懂 EgoUseBench

佩戴者看到的画面、用户提出的要求、智能体改变的网页状态，可以在不同时间点更新。短片展示 EgoUseBench 如何把这三条轨道分开，以及如何用最终网页状态判分。

https://github.com/user-attachments/assets/74b3cdaf-bd09-463a-b64f-fd1e3c26fced

## 快速开始

使用 [uv](https://docs.astral.sh/uv/) 和 Python 3.11+。不需要任务数据或模型凭据，就能检查任务定义并预览项目网站：

```bash
uv sync
uv run egocua validate data/tasks
uv run python -m website
```

打开 [http://127.0.0.1:5050](http://127.0.0.1:5050)。需要换端口时，设置 `WEBSITE_PORT`。

**运行评测。** 另需配套任务画面、Chrome 和 OpenAI 兼容的模型服务。画面的获取条件和许可见[来源登记](configs/datasets.json)。将 `.env.example` 复制为 `.env`，再填写 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`。

```bash
uv run python -m egocua.study_cli --model <model> --export experiments/
uv run python -m egocua.study_cli --study-from experiments/<冻结目录> --run-all
uv run egocua inspect-run <运行目录> --view score
```

单变化任务的默认上限为 **20 步**，组合任务为 **30 步**。运行本地测试用 `uv run pytest -q`；首次运行浏览器测试前执行 `uv run playwright install chromium`。

## 基准概览

三种设置共用 **200 道任务**。Online Ego Use 另有 **93 道组合任务**，合计 **693 个任务实例**，基于 **74 小时以上**的源录像。

| 设置 | 智能体收到什么 | 任务实例 |
| --- | --- | ---: |
| **Computer-Use** | 明确的最终要求，无第一人称画面 | 200 |
| **Offline Ego Use** | 执行前给出第一人称录像与用户要求 | 200 |
| **Online Ego Use** | 执行中陆续收到第一人称画面和用户要求 | 293 |

在线任务包括 **Graft**（用户更新要求）、**Refresh**（新画面带来新证据）和 **Clarify**（智能体提问确认指代）。智能体在 **EgoMart** 购物、在 **EgoMedia** 收藏教程，或在 **EgoMaps** 标记地点。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| [`egocua/`](egocua/) | 时间线、智能体适配、执行循环与终态检查 |
| [`sites/`](sites/) | 三个自托管网站及其冻结目录 |
| [`data/tasks/`](data/tasks/) · [`data/experiments/`](data/experiments/) | 编译后的任务定义、组合任务与评测材料 |
| [`configs/`](configs/) | 评测设置与来源登记 |
| [`tools/`](tools/) · [`tests/`](tests/) | 构建、审核与渲染脚本；离线测试 |
| [`website/`](website/) | 项目网站源码 |

## 致谢

感谢 Ego2Web、HD-EPIC、CaptainCook4D、EgoOops、HoloAssist、EgoLife 和 EgoSchema / Ego4D 团队提供来源材料，也感谢独立审核者协助核查基准任务。

短片中的第一人称画面节选自 [HD-EPIC（Damen 等，布里斯托大学）](https://doi.org/10.5523/bris.3cqb5b81wk2dc2379fx1mrxh47)，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。画面已裁切，原声已移除。

## 联系

关于基准或研究合作，请发邮件联系[陈雨润](mailto:yurunchen.research@gmail.com)。
