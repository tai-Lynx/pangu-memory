# 架构说明

## 六层

```
Adapters     Python API（当前）· FastAPI · MCP          ← 接入
Context      context_for(): 预算化拼装 · 去重 · 排序      ← 给模型的那一段
Recall       recall(): 谓词意图 + 字面相似 + 时效 + why   ← 读
Maintain     resolve(): 冲突 / 取代 / 隔离 / 衰减 / 遗忘   ← 护城河
Write        extract() + apply()                         ← 写
Store        SQLite 单文件（WAL）                          ← 落盘
             Audit：每次写入与召回留痕，可回放
```

依赖方向严格单向：`Store ← Maintain ← Recall ← Context ← Memory`。所以：

- 换存储引擎只影响 `Store`
- 换抽取器只影响 `Write`（`Memory.use_extractor`）
- 冲突策略变了只改 `Maintain`

## 为什么冲突策略是核心

如果只做「写进去 + 按相似度取出来」，这个库和 sqlite-vss 没有区别。真正决定它是不是「记忆」的，是**两条互相矛盾的信息同时存在时，系统该怎么办**。

`maintain.py` 的 `resolve()` 是一个纯函数：只读、只返回决策，不写库。写库在 `apply()` 里。这个拆分让它可测试——26 个测试里 16 个直接对着决策表打。

三条判据，按顺序：

1. **这条事实还活着吗？** `t_valid_to IS NULL AND superseded_by IS NULL AND quarantined = 0`
2. **这个谓词能容许多个值吗？** `FUNCTIONAL_PREDICATES` 决定
3. **新的比旧的更可信/更新吗？** 否则隔离

## 时间怎么表示

统一 naive-UTC 的 `YYYY-MM-DDTHH:MM:SS` 字符串。这样字符串序 = 时间序，SQLite 不需要解析就能排序。

区间语义：

| 字段组合 | 含义 |
|---|---|
| `t_valid_to IS NULL`，无 `superseded_by` | 当前有效 |
| `t_valid_to` 有值 | 已结束（可能是被取代，也可能是历史锚点） |
| `superseded_by` 有值 | 被某条新事实取代，指针可回溯 |
| `quarantined = 1` | 待裁决，存储可审计但不参与召回 |

带过去锚点的事实（「小时候」「2019年」）入库时就是闭区间，因此**天然不参与**「当前值」的竞争——不需要在冲突检测里为它们写特例。

## 为什么召回不用向量（v0.1）

10,003 条记忆，纯 Python 全扫描，P95 = 54 ms。这个数字意味着加向量索引在 v0.1 是过早优化。

而且字面召回有个好处：**可解释是天然的**。「命中是因为谓词 live_at + 字面 0.62」这句话是真的，不是事后编的。向量召回的解释往往只能写「相似度 0.83」，那不算解释。

v0.3 会加本地 embedding 做混合排序，但会保留 `why` 里能说清来源的部分。

## 已知的 floor

- 抽取靠正则，召回靠字符二元组。这两处会在复杂句式和近义改写上失手。
- 单主体视角（`self`），多角色场景要扩 subject 体系。
- 谓词表硬编码在 `models.py`，多语言要外置成配置文件。

这些不是 bug，是 v0.1 明确划的线。

## 测试策略

- `tests/test_conflict.py` —— 决策表本身，16 个用例，一个分支一个
- `tests/test_recall.py` —— 召回契约：当前值唯一、历史可查、无 why 不返回、不被无关记忆污染

新加规则先写测试，因为「策略」这种东西，改一行的代价往往是三个月后的一次错误记忆。
