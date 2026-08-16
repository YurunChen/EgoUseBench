# Mock 网站真实性核查

更新时间：2026-08-14

## 结论

EgoCUA-Bench 不需要复制某个商业网站的品牌和广告，但必须保留会影响网页操作的真实结构：搜索结果、筛选与排序、详情页、商品属性、选择状态和提交状态。网站只展示有来源或明确标为 benchmark 构造的数据，不用无来源的配送、促销和库存信息制造线索。

Ego2Web 的 500 道任务中有 230 道电商任务，其中 169 道明确提到 Amazon。因此 EgoMart 以大型综合电商的搜索和购买流程为主要参考，而不是平均混合多个网站的视觉风格。Walmart、eBay 和 Etsy 只用于核对字段覆盖及不同商品形态。

## 真实网站提供什么信息

- Amazon 的搜索接口以关键词和类目检索商品，可以返回主图、标题、价格、类目、属性和变体；详情资源进一步提供品牌、颜色、尺寸、技术信息和可用报价。[Amazon SearchItems](https://webservices.amazon.com/paapi5/documentation/search-items.html)、[Amazon ItemInfo](https://webservices.amazon.com/paapi5/documentation/item-info.html)、[Amazon Variations](https://webservices.amazon.com/paapi5/documentation/use-cases/organization-of-items-on-amazon/variations/variation-dimensions.html)
- Walmart 搜索结果包含商品 ID、标题、品牌、图片、价格、条件、商品类型和类目。[Walmart Item Search](https://developer.walmart.com/us-marketplace/docs/item-search-for-the-walmart-catalog)
- eBay 搜索结果包含主图与附图、标题、价格、条件、购买方式、类目、卖家和配送信息，并支持按类目、价格、条件及属性筛选。[eBay Browse API](https://developer.ebay.com/api-docs/buy/api-browse.html)
- Etsy 的商品由标题、描述、价格、类目、图片、库存和变体组成，变体可以改变价格与数量。[Etsy Listings](https://developers.etsy.com/documentation/tutorials/listings/)

因此，EgoMart 的最小页面链路应为：

```text
关键词搜索 → 结果卡片 → 商品详情与属性 → 加入购物车 → 核对购物车 → 提交
```

不展示不能操作的假按钮，也不把同品牌的不同商品伪装成同一商品的颜色或尺寸变体。

## 数据分布依据

使用 [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html) 的商品元数据进行校准。该数据包含真实页面中的 `average_rating`、`rating_number`、`price`、图片、类目、属性和父商品 ID。

本次从 Office Products、Home and Kitchen、Toys and Games、Grocery and Gourmet Food、Tools and Home Improvement、Electronics 六类中各顺序读取 5,000 条，共 30,000 条商品。结果如下：

| 类别 | 评分中位数 | 评论量中位数 | 评论量 P90 | 价格中位数 |
|---|---:|---:|---:|---:|
| Office Products | 4.5 | 26 | 558 | $15.75 |
| Home and Kitchen | 4.4 | 32 | 997 | $22.99 |
| Toys and Games | 4.5 | 32 | 537 | $18.99 |
| Grocery and Gourmet Food | 4.4 | 29 | 848 | $18.69 |
| Tools and Home Improvement | 4.4 | 25 | 572 | $25.50 |
| Electronics | 4.2 | 22 | 620 | $21.98 |

评论量呈明显长尾分布。原实现把缺失评论量近似均匀地放在 100 到 4,999 之间，不符合真实分布。现已改为确定性的长尾分布：P10 约为 2，中位数约为 38，P90 约为 950。任务明确使用的评分仍由 `products.json` 中的冻结值和来源说明决定，不被默认分布覆盖。

## 当前商品图片核查

EgoMart 当前有 705 个商品和 705 张互不重复的主图，所有图片都有本地路径、来源记录和内容哈希。图片统计如下：

- 98.4% 的图片最长边至少 500px；
- 72.2% 的图片最长边至少 1000px；
- 68.5% 的图片宽高比位于 0.75 到 1.33 之间；
- 约 76% 的图片四角为白色或透明背景。

Amazon 的主图规范强调白色背景、清晰完整的商品和足够分辨率；ABO 则提供真实 Amazon 商品的标题、品牌、类型、颜色、尺寸、材质和目录图片。[Amazon 商品图片说明](https://sellercentral.amazon.com/seller-forums/discussions/t/84f36e73-1fbe-4c22-9e7c-8345cc3e24e9)、[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)

当前图片满足冻结研究环境的最低要求。非白底图片保留其真实零售或制造商来源，不通过生成图或从第一人称视频裁剪来统一外观。网页使用固定方形容器和 `object-fit: contain` 展示，不拉伸或裁掉商品。

## 已落实的约束

1. 搜索页只展示主图、标题、评分、评论量、价格和可进入的详情页。
2. 详情页只展示数据中真实存在的品牌和属性；没有父商品关系时，不生成假变体。
3. 删除所有商品统一出现的 Prime、免费配送、固定到货日期和 Buy Now 等无来源状态。
4. 筛选栏只使用品牌、商品类型、颜色、尺寸、材质和格式，不把内部注释或冗长视觉描述当筛选项。
5. 页面中的按钮都对应真实状态变化；没有实现的功能不显示。
6. 自动测试持续检查价格、评分、评论量和主图质量是否偏离已登记的分布范围。

## 任务难度与补充要求核查

网页任务的难度不能只用最终步数定义。WebArena 用可完整执行的网站和程序化最终状态检查任务是否完成；WorkArena++ 则区分单一操作与需要组合多个子目标的任务。因此，本项目只把以下会改变决策的因素计入任务难度，不用验证码、无关页面或重复点击人为增加步数。[WebArena](https://webarena.dev/static/paper.pdf)、[WorkArena++](https://proceedings.neurips.cc/paper_files/paper/2024/file/0b82662b6c32e887bb252a74d8cb2d5e-Paper-Datasets_and_Benchmarks_Track.pdf)

- 初始画面或网页中有多少个合理候选；
- 决定性信息在搜索结果页还是详情页；
- 是否需要比较多个网页实体；
- 是否需要删除旧选择、加入新选择或保留多个目标；
- 是否需要填写路线、来源或其他结构化字段；
- 动态信息到达后需要保持、缩小还是更新目标。

当前只对3道核心种子生成任务。脚本正确路径均为4个交互回合；这只证明网站和检查器可以完成，不代表真实 Agent 只需4步。实际难度要用固定模型和独立人类的轨迹报告，包括搜索次数、打开详情页数量、选择修改次数和纠错次数。

当前只有1道 Graft 核心种子：用户操作割草机线头时，补充要求教程同时覆盖重新装线。它不使用价格、评分或热度制造变化。后续扩展时再检查补充要求是否过度集中；在只有1道任务时不报告模板相似度。

## 不同网站可以自然补充什么

| 网站 | 真实页面可提供的关键元素 | 适合中途补充的要求 |
|---|---|---|
| 电商 | 价格、运费、评分、评论量、品牌、规格、尺寸、颜色、套装、兼容关系 | 限价、比较总价、增加评分或评论量门槛、指定规格、要求兼容配件 |
| 视频 | 缩略图、标题、频道、发布时间、时长、观看量、字幕、语言、简介和章节 | 指定官方频道、时长、字幕或语言、发布时间、必须覆盖的步骤 |
| 地图 | 地点照片、地址、坐标、评分、营业时间、无障碍信息、路线时间与距离 | 改变交通方式、增加途经点、限制路线、要求营业中或无障碍入口 |

YouTube 官方接口明确提供缩略图、频道、发布时间、时长、字幕状态和统计数据，并支持按时长与字幕筛选。[YouTube Videos](https://developers.google.com/youtube/v3/docs/videos)、[YouTube Search](https://developers.google.com/youtube/v3/docs/search/list) Google Places 提供地址、坐标、照片、评分、营业时间和无障碍等地点字段；Routes API 提供交通方式、距离、时间、途经点和避开收费道路等路线设置。[Google Places](https://developers.google.com/maps/documentation/places/web-service/reference/rest/v1/places)、[Google Routes](https://developers.google.com/maps/documentation/routes/reference/rpc/google.maps.routing.v2)

补充条件只有在页面真实展示、Agent 可以操作、检查器能够验证时才进入任务。不能为了平衡数量给教程任务随意增加“官方频道”或“带字幕”等原数据不支持的条件。

## 边界

上述结论只证明购物网站的展示与数据尺度合理，不证明每道题的目标和干扰项都合理。每道任务仍需在人工审核页中同时核对第一人称画面、网站候选和标准答案。

当前核心 EgoMedia 任务的三个教程候选都已补齐独立缩略图、频道、发布时间、时长和观看量。当前核心集合没有 EgoMaps 任务；地图来源只有在地点照片、坐标、入口和路线数据完整时才会生成任务。

本轮另排除了4道来源与网页任务不一致的题：将 YouTube 教程改成商品购买的2道、将商品配件改成教程筛选的1道，以及不需要网页操作的电影年份查询1道。被排除记录保留在来源清单中，但不再生成可执行任务。
