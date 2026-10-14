# Verifiable Trajectory Sorted-List Baseline (Chengdu / Xi'an)

这是一个完全独立的新项目，不依赖原 τ-MMR 项目代码。

## Baseline 定义

### 空间
- 二叉递归空间划分。
- 顶点划分权重来自相邻道路上的不同轨迹数量；同一轨迹重复经过同一道路只计一次。
- 每个内部节点只有 `left` / `right` 两个递归子空间。
- 横跨两个子空间的道路保存在当前节点的普通 `cross_edges` 列表。
- `cross_edges` 不保存聚合 MBR。
- 当前空间节点被访问时，所有 `cross_edges` 必须逐条执行空间相交判断。

### 时间：Sorted Authenticated Entry List
每条道路维护：

```text
Entry = (trajectory_id, start, end)
```

索引构建阶段每插入一条 Entry 时，根据 `start` 找到其有序位置并直接插入；不存在构建结束后的统一排序步骤。除此之外不建立 `min_start/max_end`、interval tree、MMR、时间桶或时间子树。

查询 `[qs, qe]` 时：

1. 通过有序 `start` 找到右边界 `k = upper_bound(start <= qe)`；
2. `Entry[0:k]` 全部展开到 VO；
3. 其中即使 `end < qs`、最终不是查询结果，也仍必须进入 VO、参与客户端哈希恢复；
4. 若 `k < n`，额外返回 `Entry[k]` 作为第一个被排除 Entry 的右边界见证，客户端检查其 `start > qe`；
5. `Entry[k+1:n]` 不展开，仅由一个认证 suffix hash-chain state 表示；
6. 客户端用 suffix state + boundary Entry + 全部前缀 Entry 恢复完整 edge list root。

因此固定空间范围下，时间窗口增大时，`start <= qe` 的前缀通常增大，Server Query、VO、Verification 也会自然增长。

## 认证 List commitment

不是 MMR，也不是带时间摘要的 Merkle 时间树。每条 edge 使用反向顺序 hash-chain：

```text
state_n = LIST_TAIL
state_i = H(Entry_i || state_{i+1})
root_e  = state_0
```

构建 `root_e` 时，该 edge 的**所有 Entry**都会参与哈希，无论未来某次查询中它是结果还是非结果。

查询证明前缀时，可以直接携带预计算的 `state_{k+1}` 作为后缀摘要，然后客户端从后向前恢复 `state_0`。

## 与 Ours 的核心差别

| 模块 | Baseline | Ours |
|---|---|---|
| 空间树 | Binary | left + mid + right |
| cross-edge 聚合 MBR | 无 | 有 |
| cross-edge 查询 | 逐边检查 | 可先整批剪枝 |
| edge 时间结构 | start-sorted authenticated linear list | τ-MMR |
| 时间剪枝依据 | 仅 `start` 的单一右边界 | 子树 `min_start/max_end` |
| 对 `end < qs` 的前缀 Entry | 必须展开并验证 | 可被子树时间摘要整体剪枝 |
| VO | 前缀全部 Entry + 1 个边界见证 + suffix digest | 结果相关 Entry + 被剪枝子树证明 |
| 客户端 | 哈希并检查整个展开前缀 | 验证展开 Entry 与多层剪枝证明 |

## 成都正式实验参数

当前成都配置与 Ours benchmark 对齐：
- `BASE_QUERY_START = 1538502351`
- 空间：1/2/3/4/5 km
- 时间：1min/30min/1h/4h/12h/24h
- warmup = 1
- repeats = 30
- 正式对比应复用 Ours 的 `chengdu_selected_dense_region.csv` 中相同 `center_lon/center_lat`。

## 使用

先编辑 `configs/chengdu.json` / `configs/xian.json`，确保 node、edge、trajectory 路径与 Ours 使用完全相同的数据一致。

### 1. 重新构建索引

**v2 结构与旧版不兼容，必须重新 build。**

```powershell
python build_index.py --config configs/chengdu.json
```

Index Construction Time 包括：
- 每条 Entry 根据 `start` 查找位置并有序插入；
- 全部 Entry 的认证 list hash-chain 构建；
- Binary Spatial Tree 构建与认证。

原始轨迹文件读取/解析和 Catalog 准备不计入 Index Construction Time；逐 Entry 的时间位置查找与有序插入计入 Index Construction Time。

### 2. 正式 5×6 benchmark

```powershell
python benchmark_cross_range.py `
  --config configs/chengdu.json `
  --setup "E:\MMR_Trajectory_range\benchmark_output\chengdu_selected_dense_region.csv"
```

`Query Time = Server Query + Fine Filtering`，Verification Time 独立统计。

### 3. 正确性测试

```powershell
python correctness_test.py --config configs/chengdu.json --min-lon ... --min-lat ... --max-lon ... --max-lat ... --start ... --end ...
```

### 4. 独立自测试

```powershell
python self_test.py
```

## Benchmark 诊断指标

- `checked_cross_edges`
- `checked_leaf_edges`
- `spatial_matched_edges`
- `scanned_entries`：真正执行时间 overlap 检查的前缀 Entry 数
- `boundary_witness_entries`：每条有后缀的空间命中 edge 最多 1 个右边界 Entry

这些指标可用于解释随着时间范围增大，Baseline 的前缀扫描和 VO 为什么增大，以及 τ-MMR 为什么仍能通过双边时间摘要获得更细粒度剪枝。
