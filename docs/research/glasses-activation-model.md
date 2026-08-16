# EgoCUA-Bench 的眼镜感知边界

## 结论

EgoCUA-Bench 不应假设消费级智能眼镜在后台全天录像或持续把视频传给 Agent。更准确的使用场景是：**眼镜可以长期佩戴，但视觉 Agent 由用户主动唤起，并只在一次有明确起止的任务会话中获取画面。**

这一区分很重要：`always available` 不等于 `always sensing`。

## 产品证据

- Ray-Ban Meta 的视觉问答以“Hey Meta”发起；后续问题可以在同一会话中继续。Meta 也介绍了视频驱动的连续实时帮助，但前提仍是用户先启动交互，而不是后台全天感知。[Meta AI with Vision](https://about.fb.com/news/2024/04/new-ray-ban-meta-smart-glasses-styles-and-meta-ai-updates/)；[Meta 后续对话与实时视频](https://about.fb.com/news/2024/09/ray-ban-meta-glasses-new-ai-features-and-partner-integrations/)
- Meta Ray-Ban Display 明确将交互描述为用户控制的短时交互，并说明显示并非始终开启。[Meta Ray-Ban Display](https://about.fb.com/news/2025/09/meta-ray-ban-display-ai-glasses-emg-wristband/)
- Google 的 Android XR 眼镜原型有摄像头和麦克风，支持 Gemini 理解眼前环境；产品描述使用“一次点击”或语音命令启动帮助。这里的“全天帮助”表示设备适合全天佩戴和随时调用，不足以证明摄像头全天采集。[Android XR](https://blog.google/products-and-platforms/platforms/android/android-xr/)；[Android XR 眼镜演示](https://blog.google/products-and-platforms/platforms/android/android-xr-glasses-demo-io-2025/)
- Project Aria 确实采集连续的第一人称视频、音频和位置，但它是研究采集设备，不是消费产品；录制时有指示灯，参与者可以随时停止录制。[Project Aria 隐私说明](https://about.fb.com/news/2020/09/privacy-matters-project-aria/)

因此，已有产品支持两种合理形态：用户触发的一次视觉采集，以及用户开启的短时实时视觉会话。没有充分依据把后台全天视觉采集作为当前产品的默认交互。

## Ego2Web 与真实交互的区别

[Ego2Web](https://ego2web.github.io/) 把预先录制的完整第一人称视频与网页任务配对。完整视频是离线 Benchmark 输入和数据来源，并不表示真实用户会让消费级眼镜持续录制，再把整段日常视频交给网页 Agent。

EgoCUA-Bench 可以从连续视频中确认真实事件，但给 Agent 的输入必须按拟模拟的产品交互重新定义：任务开始时的短时视觉片段，或用户明确开启的任务级实时会话。数据集怎样录制，不能直接等同于产品怎样工作。

## 对三类任务的影响

| Track | 合理的真实交互 | 是否需要持续视觉 |
|---|---|---|
| Graft | 用户展示对象并发起任务；网页执行中又补充一句要求 | 不需要，后续只需语音或文本 |
| Clarify | 用户展示对象并发起任务；Agent 追问，用户回答 | 不需要，除非用户回答时重新指向对象 |
| Refresh | 用户明确开启短时实时视觉会话；会话内后来出现 V1 | 需要，但只在该任务会话内，不是全天开启 |

当前 `Graft` 和 `Clarify` 与主流产品形态一致。`Refresh` 的 V1 目前在网页首次出现搜索结果时自动释放；这个触发点便于实验复现，却不是自然的摄像头启动理由。若不声明用户已经开启实时视觉会话，它会隐含 always-on 假设。

## 推荐的统一场景

首版应统一描述为“用户主动发起的、任务级智能眼镜会话”：

1. 用户通过语音或按键启动任务。
2. 眼镜采集一个短视觉片段 V0；关键帧只是对该片段的模型输入表示，不表示用户连续拍了八张照片。
3. Agent 开始网页操作。
4. Graft 接收用户后续语句；Clarify 完成一次问答；Refresh 只在用户已经开启的实时视觉会话中接收 V1。
5. 网页任务结束后，视觉会话停止。

项目的科研问题因此不是“全天监控能否发现事件”，而是：**在一次用户主动发起的现实到网页任务中，当新的用户信息、视觉证据或澄清答案到来时，Agent 能否修正正在执行的网页目标。**

首版仍只能声称评估“Agent 对基准提供的新证据作出反应”，不能声称评估自主事件发现或连续视频理解。
