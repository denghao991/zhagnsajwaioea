# Chunk Size 评估设计

**目标：** 在现有分块策略（MarkdownHeaderTextSplitter #/## + RecursiveCharacterTextSplitter）下，
找出最适合实际文档特征的 chunk_size。

## 评估方法

直接用统计指标衡量分块质量，无需人工标注 QA pairs：

| 指标 | 含义 | 判定 |
|------|------|------|
| 总 chunk 数 | 文档被切成了多少片 | 越少越好（上限由 embedding API 费用决定） |
| token 中位数 | 一半 chunk 小于此值 | 反映典型 chunk 大小 |
| 75%/90% 分位 | 大 chunk 的分布情况 | 接近 chunk_size 是正常利用 |
| 短 chunk 占比（< 50 tokens） | 碎片化程度 | < 10% 为优，> 20% 说明偏大 |
| 超限占比（tokens > chunk_size） | 被迫截断的情况 | 应为 0%（RecursiveCharacterTextSplitter 保证） |
| 平均填充率（token 中位数 / chunk_size） | 空间利用率 | 40-70% 合理，< 30% 偏大 |

## 候选配置

chunk_size: 128, 256, 384, 512, 768, 1024

> 注：分块策略固定为当前方案（#/## 标题切分 + RecursiveCharacterTextSplitter 处理大段），
> 仅变动 chunk_size 一个参数。

## 输出示例

```
chunk_size=128:   45 chunks | 中位数 98  | 碎片率 11.1% | 填充率 76.6%
chunk_size=256:   28 chunks | 中位数 186 | 碎片率 3.6%  | 填充率 72.7%
chunk_size=384:   20 chunks | 中位数 245 | 碎片率 0.0%  | 填充率 63.8%
chunk_size=512:   15 chunks | 中位数 297 | 碎片率 0.0%  | 填充率 58.0%
chunk_size=768:   11 chunks | 中位数 312 | 碎片率 0.0%  | 填充率 40.6%
chunk_size=1024:   9 chunks | 中位数 315 | 碎片率 0.0%  | 填充率 30.8%
```

## 决策逻辑

1. **排除** 碎片率 > 20% 的配置（文档被切太碎，语义丢失）
2. **排除** 填充率 < 30% 的配置（浪费 embedding 容量和费用）
3. 其余配置中，优先选 **更高填充率** 的（信息更密集）
4. 同时参考总 chunk 数和 embedding 费用
