# LightX2V shared pinned CPU weight offload 实验报告

- 日期：2026-09-14
- 项目：`/data/liuhongda/lightx2v_offload_opt`
- 分支：`lightx2v-hunyuan-v18.0`
- HEAD：`a463566528a487c112e1ac5d65339aa3c8c2516d`

## 1. 结论

本轮重新执行了静态测试、9 组 CUDA 冒烟、9 组真实 Wan 权重初始化，以及 5 组完整 I2V 推理。对这些 artifact 的机器可读一致性审计共 **1208/1208 assertions 通过**；它们不是 1208 个相互独立的测试 case。

核心结论：

1. 共享实现确实消除了多 SP rank 的 DiT block pinned CPU 物理副本，而不是只让多个 Python 对象指向 N 份内容相同的 tensor。
2. SP8 `host` 只保留一份 15.061 GiB arena；相对 8 份 rank-private block 权重，结构上节省 105.428 GiB（87.5%）。
3. SP8 `auto`/`numa` 在本机两个 NUMA 节点上保留两份，共 30.122 GiB；结构上节省 90.367 GiB（75%）。
4. 端到端进程树峰值 PSS 的采样近似值，从 SP8-private 的约 247.942 GiB 降到 host 的约 83.776 GiB、numa 的约 97.955 GiB、auto 的约 98.636 GiB。
5. `host` 内存最少，但本次 rank0 的 40 步 transformer 合计从 private 的 46.246s 增加到 66.246s；这与跨 NUMA/共享内存带宽竞争假设一致，但本轮未单独采集 H2D、PCIe/UPI 或内存控制器计数器，不能据此确定因果。
6. `numa`/`auto` 的 40 步合计分别为 44.014s/43.801s；本次单次实验没有观察到类似 host 的阶段时延增长，但不足以证明无 H2D 退化，模型加载阶段也更长。
7. SP8 private、shared-host、shared-numa、shared-auto 的 MP4、逐帧 RGB 与 frame MD5 全部完全一致。
8. 每卡显存峰值几乎不变（约 15.3 GiB），符合“只共享 CPU block staging weights”的设计边界。

## 2. 实验环境

| 项目 | 实测值 |
|---|---|
| GPU | 8 × NVIDIA H200，143771 MiB/卡 |
| GPU 拓扑 | GPU 0–3 属于 NUMA 0；GPU 4–7 属于 NUMA 1；8 卡间 NV18 |
| CPU | Intel Xeon Platinum 8568Y+，2 socket，48 core/socket，SMT2，192 logical CPU |
| NUMA | 2 nodes；node0 CPUs `0-47,96-143`，node1 CPUs `48-95,144-191` |
| 物理内存 | 2,159,639,281,664 bytes，约 1.964 TiB |
| Swap | 0 |
| Python | 3.11.13 |
| PyTorch | 2.8.0+cu128 |
| PyTorch build CUDA version | 12.8 |
| transformers | 5.14.1 |
| vLLM | 0.11.0 |
| PyAV | 16.1.0 |
| FFmpeg | imageio-ffmpeg bundled 7.0.2-static |

原始环境记录：

- [CPU/NUMA](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/hardware_lscpu.txt)
- [内存](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/hardware_memory.txt)
- [GPU/NUMA 拓扑](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/hardware_gpu_topology.txt)
- [软件版本](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/software_versions.txt)

`hardware_numa.txt` 为空，因为系统没有 `numactl` 命令；本报告的 NUMA 结论来自 `lscpu`、`nvidia-smi topo -m`、GPU PCI sysfs 与运行时 `/proc/<pid>/numa_maps`，没有使用该空文件。

## 3. 固定推理条件

所有完整推理使用相同条件：

- 模型：Wan2.1-I2V-14B-720P-Lightx2v。
- DiT/T5/CLIP：`fp8-vllm`。
- DiT offload：`cpu_offload=true`、`offload_granularity=block`、`lazy_load=false`。
- 推理步数：40。
- 帧数：81。
- seed：42。
- 输入：`assets/inputs/imgs/img_0.jpg`，原图 832×1104。
- 请求目标面积：`480×832`。
- SP8：Ulysses sequence parallel，8 个 rank。
- T5、CLIP、VAE：不做 CPU offload。

Wan I2V 默认预处理保持输入图像宽高比，并在目标面积内对齐 VAE stride/patch。由原图 832×1104 与目标面积 480×832 算出的实际输出为 **544×720**，不是固定拉伸成 832×480。

## 4. 验证方法

### 4.1 四层验证

1. **静态/故障测试**：manifest、布局、SysV 生命周期、NUMA 边界、CUDA 注册失败清理、rank 错误协调、Wan checkpoint adapter、shared weight map 与监控解析。
2. **CUDA smoke**：SP1/SP4/SP8 × auto/numa/host，共 9 组；分配 32 MiB arena，检查完整 payload pinned 与异步 H2D 逐元素一致。
3. **真实 Wan 初始化**：同样 9 组；使用 40 个真实 block checkpoint，验证 1760 tensor views、leader-only populate、shmid/group、CUDA 注册、NUMA 页驻留与退出清理。
4. **完整推理**：single-private、SP8-private、SP8-shared-host、SP8-shared-numa、SP8-shared-auto，共 5 个视频。

### 4.2 GPU 选择

| world size | 物理 GPU | 目的 |
|---:|---|---|
| 1 | GPU 7 | 单 rank，位于 NUMA 1 |
| 4 | GPU 0,1,4,5 | 两个 rank/NUMA，主动覆盖跨 NUMA 分组 |
| 8 | GPU 0–7 | 每个 NUMA 四个 rank |

SP4 中 event 的 `cuda_ordinal=2,3` 是可见设备中的 local ordinal，实际对应物理 GPU 4、5；报告使用 PCI BDF 和脚本映射确定物理卡。

### 4.3 内存口径

- **精确结构量**：唯一 shmid 数 × `arena_bytes`，代表 DiT block 共享物理页。
- **实际驻留**：`numa_maps` 非加和最大页数 × kernel page size。
- **进程总内存**：进程树 `smaps_rollup` 的 PSS 求和。每次 sweep 按 PID 依次读取，不是原子快照。
- **不采用 sum RSS 作为物理量**：同一共享页会在每个进程 RSS 中重复出现。
- **峰值 PSS** 包括 checkpoint 加载临时 tensor、Python/runtime、non-block 等；它不能全部归因于稳定态 block arena。只有纳入某 shared segment 的所有 mapper 时，该映射的 sum PSS 才近似其物理驻留。
- 每个 CSV 峰值的 PSS/RSS/private/shared 均取同一采样行，没有把不同时间的列峰值拼接。
- SP8-private 有效相邻采样的最大间隔为 15.497s；因此全进程树峰值是近似值，不用几 GiB 的小差异排序。

## 5. 自动化检查

| 检查 | 结果 |
|---|---:|
| Pytest | 69 passed，0 failed，0 error，0 skipped |
| Ruff / Python bytecode compile | passed |
| Bash syntax / `git diff --check` | passed |
| artifact 一致性断言 | 1208 passed，0 failed |
| CUDA smoke settings | 9/9 passed |
| 真实 Wan init settings | 9/9 passed |
| 完整视频 | 5/5 成功并可完整解码 |
| 退出后目标 SysV segment | 0 个残留 |

静态 pytest 运行于当前受限子进程，产生 1 条 `cudaGetDeviceCount error 304` warning，但无 skip/fail；真实 CUDA 能力由独立 H200 smoke 和 E2E 覆盖。完整控制台记录见 [pytest_shared_offload.log](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/pytest_shared_offload.log)。

机器可读总结果：[shared_offload_summary_20260914.json](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_summary_20260914.json)

## 6. CUDA smoke 结果

每组均满足：完整 32 MiB payload H2D 正确、每个 group 只有 leader populate、同组 shmid 相同、不同 group shmid 不同、注册覆盖完整 arena。

| SP | scope | groups | wall time | H2D |
|---:|---|---|---:|---|
| 1 | auto | `[0]` | 4s | ok |
| 1 | numa | `[0]` | 5s | ok |
| 1 | host | `[0]` | 5s | ok |
| 4 | auto | `[0,1]`, `[2,3]` | 9s | ok |
| 4 | numa | `[0,1]`, `[2,3]` | 10s | ok |
| 4 | host | `[0,1,2,3]` | 9s | ok |
| 8 | auto | `[0,1,2,3]`, `[4,5,6,7]` | 14s | ok |
| 8 | numa | `[0,1,2,3]`, `[4,5,6,7]` | 15s | ok |
| 8 | host | `[0,1,2,3,4,5,6,7]` | 14s | ok |

## 7. 真实 Wan 权重初始化

一份 arena 的精确大小是 `16,171,827,200 bytes = 15.061187744 GiB`，包含 40 个 block 文件中的 1760 个 tensor view；每 rank 使用 120 个 tensor-aware CUDA registration 区间。

下表“构造区间”是从第一个 rank 开始加载 private non-block 到最后一个 rank 完成 1760 views 校验；“wall”包括 Python/import/torchrun 和人为设置的 15 秒观察窗口。

| SP | scope | arena groups | 物理 block arena | block 节省率 | 构造区间 | wall 含 hold | 进程树峰值 PSS |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | auto | 1 | 15.061 GiB | 0% | 11.175s | 43s | 17.766 GiB |
| 1 | numa | 1 | 15.061 GiB | 0% | 10.707s | 44s | 17.736 GiB |
| 1 | host | 1 | 15.061 GiB | 0% | 11.802s | 44s | 17.729 GiB |
| 4 | auto | 2 | 30.122 GiB | 50% | 37.064s | 78s | 39.682 GiB |
| 4 | numa | 2 | 30.122 GiB | 50% | 33.561s | 76s | 40.162 GiB |
| 4 | host | 1 | 15.061 GiB | 75% | 30.840s | 73s | 24.624 GiB |
| 8 | auto | 2 | 30.122 GiB | 75% | 38.361s | 89s | 48.578 GiB |
| 8 | numa | 2 | 30.122 GiB | 75% | 32.283s | 82s | 48.580 GiB |
| 8 | host | 1 | 15.061 GiB | 87.5% | 29.003s | 78s | 33.508 GiB |

每个 setting 均满足：

- `WAN_SHARED_INIT_OK` 数量等于 world size。
- `Validated 1760 ... views` 数量等于 world size。
- `Populating 40 ... files` 数量等于 group 数，证明 follower 没有读取 block payload。
- `registered_bytes == arena_bytes`，registration chunks 为 120。
- 监控时所有目标 PID 存活、无 `/proc` 读取错误。
- 退出后本轮所有 shmid 均不存在。

## 8. NUMA 驻留证据

每份 arena 恰有 `3,948,200` 个 4 KiB 页，即 16,171,827,200 bytes。

- SP4/SP8 `numa` 和 `auto`：NUMA 0 group 的全部页在 node 0，policy 为 `bind:0`；NUMA 1 group 的全部页在 node 1，policy 为 `bind:1`。
- SP1 auto/numa 使用 GPU 7，全部页在 node 1。
- `host` 不执行 `mbind`，policy 为 `default`；SP4-host 本次全部页落在 node 0，而 SP8-host 本次全部页落在 node 1。

这说明 `host` 的页位置取决于 leader 首次触页时的 CPU 调度，不保证与 leader GPU 或所有 GPU 本地。SP8-host 的一个 arena 被两个 socket 的 GPU 同时读取，必然存在一侧远端访问，并可能争用同一内存控制器；它不是死锁，但可能影响 H2D 供给速度。

## 9. 完整端到端结果

运行顺序固定为：SP8-private → SP8-shared-host → SP8-shared-numa → SP8-shared-auto → single-private。每项 N=1。

| case | wall | Load models (rank0) | Run main (rank0) | 40 步合计 (rank0) | warm step 中位数 | Pipeline (rank0) | Total Cost (rank0) | 约峰值 PSS | GPU 峰值/卡 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-private | 229s | 31.923s | 182.323s | 180.163s | 4.494s | 183.941s | 220.729s | 31.156 GiB | 17550 MiB |
| SP8-private | 153s | 70.090s | 47.152s | 46.246s | 0.999s | 48.027s | 123.169s | 247.942 GiB | 15686 MiB |
| SP8-shared-host | 162s | 64.910s | 67.130s | 66.246s | 1.505s | 67.957s | 137.854s | 83.776 GiB | 15672 MiB |
| SP8-shared-numa | 153s | 82.453s | 44.977s | 44.014s | 0.944s | 45.860s | 133.330s | 97.955 GiB | 15670 MiB |
| SP8-shared-auto | 149s | 78.398s | 44.709s | 43.801s | 0.939s | 45.632s | 128.957s | 98.636 GiB | 15672 MiB |

`wall` 包含 torchrun/Python 启动和退出，由秒级 `date +%s` 记录；框架 `Total Cost` 从构建 runner 开始计时；`Pipeline` 主要覆盖一次请求。日志名为 `Run DiT` 的计时实际包住整个 `run_main()`，包含初始化、40 步、VAE decode 和 postprocess，因此表中更名为 `Run main`。`40 步合计` 是 rank0 的 40 条 `Run Dit every step` 求和；`warm step 中位数` 是第 2–40 步的中位数。

### 9.1 相对 SP8-private

| shared scope | 约峰值 PSS 变化 | wall 变化 | Load models 变化 | 40 步合计变化 | Total Cost 变化 |
|---|---:|---:|---:|---:|---:|
| host | -164.166 GiB（-66.21%） | +9s | -5.180s（-7.39%） | +20.000s（+43.25%） | +14.684s（+11.92%） |
| numa | -149.986 GiB（-60.49%） | 0s | +12.363s（+17.64%） | -2.232s（-4.83%） | +10.161s（+8.25%） |
| auto | -149.306 GiB（-60.22%） | -4s | +8.308s（+11.85%） | -2.445s（-5.29%） | +5.788s（+4.70%） |

峰值 PSS 的下降大于纯 block 稳定态的结构节省，因为 SP8-private 峰值出现在 checkpoint 加载期，还包含多 rank 的 source tensor/映射等临时内存。可靠的稳定结构结论仍应使用唯一 arena：host 精确少 105.428 GiB，auto/numa 精确少 90.367 GiB。

`numa`/`auto` 的约 2.2–2.4 秒差异不能视为确定性能提升：本轮每项仅运行一次，且存在顺序、page cache、GPU warm-up 和系统噪声。本轮只能说这两项未观察到 host 那样的阶段时延增长，不能据此宣称 H2D 无退化。

PSS 表格的三位小数用于保留原始汇总值，不表示采样具有同等精度；`wall` 只有 1s 分辨率，因此小差异不做高精度百分比解读。

## 10. 输出正确性

所有 5 个视频均满足：

- 进程退出码 0，日志存在 `Video saved successfully`。
- H.264，544×720，16 fps，81 帧。
- PyAV 完整解码 81 帧，无异常。
- FFmpeg 使用 `-xerror -err_detect explode` 完整解码，stderr 为空。
- FFmpeg framemd5 恰有 81 帧，stderr 为空。

| case | container SHA-256 | decoded RGB24 SHA-256 |
|---|---|---|
| SP8-private | `6fd97f5012b0bf98b8027b9eecb1082dfed54648a86f7d03ac32cf12e074b311` | `3332862d21e288ce42282293987a4f0fc88e7a2f356678a0688e023303f1b4bd` |
| SP8-shared-host | 同 SP8-private | 同 SP8-private |
| SP8-shared-numa | 同 SP8-private | 同 SP8-private |
| SP8-shared-auto | 同 SP8-private | 同 SP8-private |
| single-private | `329d86a3b88eb0dbf84882b18922c83d292ba0dbf0a8e5afbb41e1e708e5b62b` | `3c822582c387bb974ae5472a994c4efcd6cd33be1bf987b2a48c63762e6a9f73` |

四个 SP8 输出连容器字节都完全相同，证明共享 arena 与 scope 选择没有改变同一 SP8 算法的数值结果。single-private 与 SP8 的并行归约/运算顺序不同，不能要求逐位一致；它独立通过完整解码，抽查第 40 帧也得到符合 prompt 的猫、墨镜和水面场景。

视频位置：

- [SP8 private](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_e2e_20260914/sp8_private/output.mp4)
- [SP8 shared host](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_e2e_20260914/sp8_shared_host/output.mp4)
- [SP8 shared numa](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_e2e_20260914/sp8_shared_numa/output.mp4)
- [SP8 shared auto](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_e2e_20260914/sp8_shared_auto/output.mp4)
- [single private](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_e2e_20260914/single_private/output.mp4)

## 11. 数据分析

### 11.1 是否解决 N 倍 CPU block 权重

是。SP8-private 的 block 结构参考值是 `8 × 15.061 = 120.490 GiB`。实测 shared-host 只有一个 shmid/15.061 GiB，shared-numa/auto 只有两个 shmid/30.122 GiB；每份的 PSS、驻留页数和 arena bytes 精确一致。

每个 rank 仍显示一段 15.061 GiB 虚拟映射，且每 rank 都注册 15.061 GiB，这不代表物理页有 N 份。PSS 对同一共享页按映射数分摊后求和，回到一份或两份 arena 的精确大小。

### 11.2 host 为什么慢

Host scope 只让一个 leader 读取 40 个 block，本次 Load models 比 private 快 5.18s；但 host 是第二个 E2E case，page cache 可能已变热，所以不能把加载差异完全归因于 leader 数量。推理时，这份 arena 的页本次全部驻留在 NUMA 1，NUMA 0 GPU 访问会跨 socket，而 8 个 rank 会并发从同一份页预取 block。观察到的 40 步合计增加约 20s、warm step 中位数从 0.999s 增加到 1.505s，与跨 NUMA/共享带宽竞争假设一致，但尚需 H2D、PCIe/UPI 和内存控制器计数器才能验证因果。

### 11.3 numa/auto 为什么是推荐平衡点

两个 NUMA 各自拥有一份本地 arena：

- 物理 block 页仍比 private 少 75%。
- 每个 arena 只服务四个本地 GPU。
- 避免跨 socket H2D。
- 本次 40 步合计与 private 同一量级，没有观察到 host 的时延增长。

本次 Load models 比 private 高 8–12s。这与两个 leader 填充权重、以及所有 rank 的 checkpoint key/signature preflight 引入的一次性工作相符，但运行顺序/page cache 也混入了该差异，不做单一因果结论。对长视频、多请求或常驻服务，这类一次性初始化成本更容易摊薄。

### 11.4 为什么显存没有下降

共享发生在 CPU staging 层。每 rank 的两个 GPU block slot、sequence-parallel activation，以及完整 T5/CLIP/VAE 权重仍是私有的。SP8 四项每卡峰值仅相差 16 MiB，属于采样/allocator 级差异，不能解释为 shared scope 带来显存优化。

### 11.5 建议

- 常规双/多 NUMA 服务器：使用 `shared_cpu_weight_scope=auto`。
- 确定拓扑可靠且希望错误尽早暴露：可显式使用 `numa`。
- CPU 容量是绝对瓶颈、能接受吞吐下降：使用 `host`。
- 上线前按真实并发重复至少 5 次，分别统计冷 cache、热 cache、长请求和多请求吞吐；本轮 N=1 只用于功能与数量级判断。

## 12. 实验局限

- 每个 E2E case 只运行一次，没有均值、方差和置信区间。
- case 串行执行，后运行项可能受文件 page cache 与 GPU warm-up 影响。
- 进程树 PSS 采样需要读取大量 `/proc/*/smaps_rollup`；私有 SP8 内存很大时，单轮采样会显著变慢，本轮有效相邻采样最大间隔 15.497s，可能错过短暂阶段。
- 峰值 PSS 是全进程树，不只包含 DiT block；结构量与全进程峰值必须分开解释。
- 当前没有压力叠加其他 CPU/GPU 作业；结论不能直接代替生产并发带宽测试。
- T5、CLIP、VAE 和 non-block 尚未纳入 CPU 共享。

## 13. 验证过程中修正的测试工具问题

两项问题都属于实验工具，不是 offload 推理失败：

1. 多 rank 并发 stdout 偶尔把多个标记粘在同一行。原观察脚本按“行数”计数会漏计，已改为按 marker 出现次数；监控 JSON parser 也改为全文逐 marker 解码，并新增回归测试。原始 matrix 是修正解析器前生成的；修正后重跑了 69 项静态测试，并用新解析器重新审计了全部原始日志，没有重跑已成功的 9+9 matrix GPU 任务。
2. 首个 SP8-private 已成功生成视频，但初版 validator 把请求面积误当成固定输出尺寸，按 832×480 断言失败。修正为 Wan 实际对齐尺寸 544×720 后，复用本轮刚生成的视频完成解码/哈希，没有使用历史结果。

修正工具后，对全部已生成 artifact 重新执行了汇总审计；未重跑已成功的推理。E2E 其余 case 使用修正后的校验器。

## 14. 原始数据与复现材料

- 使用与代码接入报告：[shared_offload_usage_report_20260914.md](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_usage_report_20260914.md)
- 机器可读汇总：[shared_offload_summary_20260914.json](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_summary_20260914.json)
- 报告/汇总/视频等核心 artifact 哈希：[shared_offload_artifacts_20260914.sha256](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_artifacts_20260914.sha256)
- 9×smoke + 9×Wan init：[shared_offload_matrix_20260914](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914)
- 5×E2E：[shared_offload_e2e_20260914](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_e2e_20260914)
- Pytest JUnit：[pytest_shared_offload.xml](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/pytest_shared_offload.xml)
- Pytest 完整输出：[pytest_shared_offload.log](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/pytest_shared_offload.log)
- 汇总脚本：[summarize_results.py](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/summarize_results.py)
- 测试矩阵脚本：[run_scope_world_matrix.sh](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/run_scope_world_matrix.sh)
- E2E 脚本：[run_e2e_matrix.sh](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/run_e2e_matrix.sh)
- 实现快照文件列表：[implementation_snapshot_files.txt](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/implementation_snapshot_files.txt)
- 实现文件 SHA-256：[implementation_files.sha256](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/implementation_files.sha256)
- 包含 tracked/untracked 实现的快照：[implementation_worktree_snapshot.tar.gz](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/implementation_worktree_snapshot.tar.gz)
- 快照归档 SHA-256：[implementation_worktree_snapshot.tar.gz.sha256](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/implementation_worktree_snapshot.tar.gz.sha256)
- 实验时工作树状态：[git_status_final.txt](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/git_status_final.txt)
- tracked diff：[tracked_changes.patch](/data/liuhongda/lightx2v_offload_opt/save_results/shared_offload_matrix_20260914/tracked_changes.patch)

当前实现包含尚未提交的 tracked/untracked 文件。`git_status_final.txt`、`tracked_changes.patch` 和文件哈希可审计文件身份，但 tracked diff 不包含 untracked 内容，不能仅凭这三项从 HEAD 完整重建实现。上述快照归档包含文件列表中的 tracked/untracked 实现与测试文件，可解压到该 HEAD 上重建本轮实现内容；正式复现仍建议把实现提交到 Git。
