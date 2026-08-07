# Hunyuan Image 3 T2I 分阶段混合并行实现与实验报告

本文记录 Hunyuan Image 3.0 80B 在单机 4 张 NVIDIA H200 上的分阶段并行实现、权重组织、代码改动和实机验证结果。目标是在同一次进程生命周期、同一份常驻 Transformer 权重上完成：

- AR 阶段：TP4；
- denoise 阶段：TP2 + SP2；
- 权重从官方 safetensors 直接读取，不使用离线转换工具；
- 常驻权重保持 `ABAB`，阶段切换不重新读取、不跨卡搬运、不重建模型。

实验仓库为 `/data/liuhongda/LightX2V`，官方权重目录为 `/data/liuhongda/HunyuanImage-3-Instruct`。本文数据采集日期为 2026-08-06 至 2026-08-07。

## 1. 结论

方案已经在 4 张 H200 上完整跑通。最终实现满足以下核心约束：

1. 四个进程只在初始化阶段直接读取官方的 32 个 safetensors 分片。没有新增 checkpoint 格式，也没有离线转换步骤。
2. 对参与 TP 切分的线性层，storage TP 固定为 2。rank 0/2 常驻 A，rank 1/3 常驻 B，形成 `ABAB`；norm、router 等非 TP 参数仍在四卡复制。
3. A、B 各自包含两个更细粒度的 micro-shard。AR 时四个物理 rank 分别选择 `a1、b1、a2、b2` 的零拷贝视图，并使用 WORLD 进行 TP4 collective。
4. denoise 时每个 rank 重新使用完整的 A 或 B 常驻视图；TP 组为 `[0,1]`、`[2,3]`，SP 组为 `[0,2]`、`[1,3]`。
5. 两次完整实测中，AR 到 denoise 的日志切换间隔约 12–15 ms，中间没有 Transformer checkpoint reload 或权重迁移。
6. 50-step、1024×1024 T2I 完整推理成功，输出为清晰、语义正确的 RGB PNG。
7. 用户给出的原始 8 卡 SH/JSON 保持不变；4 卡分阶段方案使用独立的新文件名，旧任务不会被新配置覆盖。

这里的“阶段切换零搬运”专指 AR/denoise 切换时不复制或重排权重。官方 gate/up 权重和 FlashInfer expert 权重仍需要在初始化或首次使用时做一次内存内组织；这一步不是离线转换，且构造完成后整个推理生命周期只保留一套通用常驻布局。

## 2. 硬件与软件环境

本机共有 8 张 H200，本实验固定使用 GPU 0、1、2、3。

| 项目 | 实验值 |
| --- | --- |
| GPU | 8 × NVIDIA H200；实验使用前 4 张 |
| 单卡显存 | 143771 MiB（约 140.4 GiB） |
| GPU 互联 | 任意两卡之间均为 NV18 |
| Driver | 565.57.01 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| NCCL | 2.27.3 |
| FlashInfer | 0.6.13 |
| 权重类型 | 官方未量化 safetensors，BF16 推理 |

选择 GPU 0–3 的原因是四张卡位于同一 CPU/NUMA 节点，并且彼此具有 NV18 直连，适合 WORLD TP4、TP2 和 SP2 三类 collective。

## 3. 并行与权重拓扑

### 3.1 三种 rank 概念

实现中显式区分三种 rank：

- `global_rank`：torchrun 的物理进程序号；
- `storage_tp_rank`：决定该进程常驻 A 还是 B；
- `logical_tp_rank`：决定该进程在规范 TP4 权重顺序中的位置。

四卡映射如下：

| Global rank / GPU | 常驻 storage shard | Denoise TP rank | Denoise SP rank | AR physical TP rank | AR logical TP rank | AR 使用的权重 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 / cuda:0 | A | 0/2 | 0/2 | 0/4 | 0 | a1 |
| 1 / cuda:1 | B | 1/2 | 0/2 | 1/4 | 2 | b1 |
| 2 / cuda:2 | A | 0/2 | 1/2 | 2/4 | 1 | a2 |
| 3 / cuda:3 | B | 1/2 | 1/2 | 3/4 | 3 | b2 |

因此有两种同时成立的观察顺序：

- 按物理 rank 看：`a1, b1, a2, b2`；
- 按 checkpoint 的规范连续分片看：`a1, a2, b1, b2`。

绝大多数 TP row-parallel reduction 是求和，顺序不影响语义。需要按列拼接结果的场景，例如 AR 的 vocab logits，则使用 `[0,2,1,3]` 将物理 gather 结果恢复成规范顺序。这里的 `torch.cat` 会为 logits 分配输出并复制数据，但它处理的是远小于模型参数的活动输出，不搬动模型权重。

### 3.2 静态创建的通信组

DeviceMesh 使用 `[seq_p, tensor_p] = [2,2]`：

```text
mesh = [[rank 0, rank 1],
        [rank 2, rank 3]]
```

进程组在初始化时一次性创建，阶段切换只选择已有组：

| 用途 | 进程组 |
| --- | --- |
| storage TP2 | `[0,1]`、`[2,3]` |
| AR TP4 | WORLD，即 `[0,1,2,3]` |
| denoise TP2 | `[0,1]`、`[2,3]` |
| denoise SP2 | `[0,2]`、`[1,3]` |

AR 不启用 SP；denoise 的两个 SP rank 分别持有同一 storage shard 的副本，所以 SP 交换序列/attention head 时不会破坏 TP shard 对齐关系。

### 3.3 阶段状态机

运行流程如下：

```text
torchrun 初始化 4 rank
        |
        v
一次性创建 WORLD / TP2 / SP2 进程组
        |
        v
每个 rank 直接读取官方 checkpoint -> 常驻 ABAB
        |
        v
activate_phase("ar") -> WORLD TP4 -> 文本思考/重写与 AR token 生成
        |
        v
CUDA synchronize + WORLD barrier
        |
        v
activate_phase("denoise") -> TP2 + SP2 -> serial CFG diffusion
        |
        v
rank 0 加载 VAE decoder -> 解码和保存 PNG
```

每次 `activate_phase()` 首先创建两个很小的 int32 状态 tensor，通过 WORLD `all_gather_into_tensor` 核对所有 rank 的 current/target phase；即使本地判断为 no-op 也会执行这次共识。真正发生 phase 变化时，再执行 CUDA synchronize 与 WORLD barrier，避免某个 rank 已进入下一阶段 collective，而另一个 rank 仍有上一阶段 kernel 排队。它不创建权重 tensor、不创建进程组，也不修改 weight storage。

## 4. 官方权重直接加载

### 4.1 没有离线转换

本实现继续使用原有的官方 checkpoint 入口：每个 rank 通过 `safe_open()` 依次打开 `model-0001-of-0032.safetensors` 到 `model-0032-of-0032.safetensors`。当前 Safetensors API 路径会先由 `get_tensor()` 在 CPU 侧物化当前 key 的完整 tensor，再按 `storage_tp_rank` 选择并 `contiguous()` 出 TP2 shard，最后转换到目标设备和 dtype。这里的“直接加载”指直接消费官方文件格式，不表示磁盘层能只读取 A/B 对应的字节范围。

关键边界是：

- 磁盘输入始终是官方权重；
- 没有预生成 A/B 文件、a1/a2/b1/b2 文件或新的 index；
- 没有新增转换命令或部署前处理步骤；
- 四个 rank 都直接读取官方文件并逐 key 取得完整 CPU tensor，因此启动 I/O 和瞬时 host memory 较重，但部署路径最简单、最可靠；
- 所有 Transformer shard 只在 AR 之前加载一次。

### 4.2 ABAB 常驻

对 `_tp_split_type()` 识别出的 TP 线性权重，storage TP 大小固定为 2：

- `storage_tp_rank=0` 保留 checkpoint 的 A 半片；
- `storage_tp_rank=1` 保留 checkpoint 的 B 半片；
- SP rank 0 和 1 分别持有相同 A/B 分片的副本。

物理结果就是：

```text
rank:     0    1    2    3
storage:  A    B    A    B
```

每个 A/B 内部保留两个连续 micro-shard。AR 选择其中一个，denoise 使用两个；二者引用同一底层 storage。

`ABAB` 不是“模型中所有 tensor 都只剩一半”的简称。RMSNorm、MoE router、embedding 以及其他未登记为 row/column/QKV/gate-up 的参数仍在每个 rank 完整复制；只有参与 TP 切分的主线性权重遵循 A/B resident 语义。

### 4.3 不同权重类型的组织方式

#### 普通 column-parallel 权重

初始化时沿输出维按 storage TP2 选择 A/B。AR 再沿同一维对 A/B 做 `narrow()`，获得 a1/a2/b1/b2；denoise 直接使用完整 A/B。

#### row-parallel 权重

初始化时沿输入维选择 A/B。AR 选择相应输入 micro-shard，并在 WORLD 上执行一次 all-reduce；denoise 使用完整 A/B，并在活动 TP2 组上执行一次 all-reduce。bias 仍复制并在 reduction 后添加。

#### 融合 QKV

Hunyuan Image 3 的官方 QKV 不是可以任意平分的普通列布局，其语义是：

```text
[kv_head, q_heads_per_kv + key + value, head_dim]
```

实现先验证 Q/KV head 的可整除性，再以完整 KV group 为单位切 storage shard 和 micro-shard，避免把同一个 KV group 从中间切开。

#### 融合 gate/up

官方顺序为：

```text
[gate_all, up_all]
```

加载时一次性组织成 storage shard 内的 micro-major 顺序：

```text
[gate_micro0, up_micro0, gate_micro1, up_micro1]
```

这样 AR 的每个 gate/up 权重都是连续 `narrow()` 视图。denoise 不恢复或移动权重，只把 projection 激活 reshape 为 `[micro, gate/up, width]` 后执行 SwiGLU，再恢复为 down projection 所需的连续 intermediate 顺序。

#### FlashInfer MoE expert

FlashInfer 要求连续的 expert pack。每个 MoE block 在首次使用时一次性构造：

```text
W1: [micro, expert, 2 * intermediate_micro, hidden]
W2: [micro, expert, hidden, intermediate_micro]
```

构造过程会先分配该 MoE block 的最终 pack，再逐 expert copy 并立即释放对应原 leaf。这避免了稳态同时常驻两套 expert 权重，并把初始化瞬时重叠限制为“最终 pack + 尚未释放的 leaves”；它并不意味着构建峰值时完全没有新旧权重重叠。之后：

- AR：选择一个 `pack[micro_id]` 连续视图，调用一次 FlashInfer；
- denoise：对 A 或 B 内的两个 micro view 各调用一次 FlashInfer，本地相加，再在活动 TP2 组上只做一次 all-reduce。

FlashInfer 始终使用规范的 logical TP4 rank 元数据。对于 A，两个调用对应逻辑 rank 0/1；对于 B，对应逻辑 rank 2/3。这样两次局部 partial sum 加 TP2 reduction 与四个 micro-shard 的全局求和等价。

## 5. T2I 业务流程

Hunyuan Image 3 T2I 不是单纯的 diffusion 调用，本次配置包含 AR 文本增强和 diffusion 两个计算特征明显不同的阶段。

### 5.1 AR 阶段

输入中文 prompt 后，runner 激活 `ar`：

1. 使用 TP4 完成 think/recaption；
2. 四个 rank 各使用一份 micro-shard；
3. attention Q/KV head 数随活动 TP 大小动态变为全局 head 数的 1/4；
4. row-parallel 输出在 WORLD 上归约；
5. lm_head 的四份 vocab logits gather 后按 `[0,2,1,3]` 拼回规范词表顺序；
6. AR 使用独立的 FlashInfer autotune cache。

本次完整实验将“汽车、打电话的驾驶员、副驾驶的狗”增强为带构图、环境、光影和海报文字的详细描述。

### 5.2 Denoise 阶段

AR 结束后，runner 激活 `denoise`：

1. 每个 rank 恢复使用完整 A 或 B view；
2. 输入序列沿 SP2 拆分；
3. Ulysses 在 `[0,2]`、`[1,3]` 内执行 seq-to-head/head-to-seq all-to-all；
4. 每个 SP 副本内部在 `[0,1]` 或 `[2,3]` 上执行 TP2 reduction；
5. CFG 使用 `serial` 模式，conditional/unconditional 分支依次执行，保证每次 SP transformer forward 的 batch size 为 1；
6. denoise 使用另一份独立 FlashInfer cache。当前实现有意隔离两阶段的调优结果，不承诺也不依赖跨 phase cache 复用；这不是对 FlashInfer cache 格式绝对不兼容的断言。

`serial CFG` 牺牲了一部分吞吐，但 4 个 rank 已全部用于 TP2×SP2，若再启用 CFG parallel 就需要扩大 world size 或改变当前业务约束。

### 5.3 VAE 与输出

50 个 diffusion step 完成后，只在输出 rank 加载独立的 VAE decoder，解码 latent 并保存 PNG。日志中的 VAE 权重加载不属于 80B Transformer 重载。

## 6. 代码改动

### 6.1 并行上下文

`lightx2v/models/networks/hunyuan_image3/parallel.py`

- 创建固定的 `[SP, TP]=[2,2]` DeviceMesh；
- 暴露 storage、AR、denoise 三类 rank/group 属性；
- 计算 physical-to-logical 映射和 gather 顺序；
- 提供同步的 `activate_phase()` 和临时 `stage()`；
- 在阶段边界记录 rank 拓扑及 CUDA allocated/reserved 显存。

### 6.2 配置归一化

`lightx2v/utils/set_config.py`

- 新增 phase-aware schema 的归一化和强校验；
- 验证 AR TP、denoise TP×SP 都覆盖 world；
- 验证 storage TP 等于 denoise TP；
- 验证 attention head、intermediate 和 vocab 等维度可被 TP4 整除；
- 将 denoise 拓扑写入旧代码使用的 flat alias，保持兼容；
- 只在 `model_cls=hunyuan_image3` 且 `phase_aware=true` 时启用新路径。

### 6.3 官方 checkpoint 内存切片

`lightx2v/models/networks/hunyuan_image3/model.py`

- 模型级 TP 字段固定描述 storage TP2，不跟随运行阶段改变；
- 继续从官方 safetensors 逐文件直接读取；
- QKV、gate/up、row/column 权重使用对应的 storage shard helper；
- SP pre/post process 改为查询活动 seq group；
- SP gather buffer key 加入 phase/group，避免不同拓扑误复用。

### 6.4 通用 TP 权重视图

`lightx2v/models/networks/hunyuan_image3/weights/hybrid_tp.py`

- 实现 phase-neutral TP linear；
- AR 用 `narrow()` 选择 micro view，denoise 直接使用 resident view；
- row reduction 动态选择 WORLD 或 TP2 group；
- 实现 grouped QKV 和 micro-major gate/up；
- 明确拒绝当前未支持的量化、LoRA 和 diff branch。

### 6.5 Dense/FlashInfer 权重

`lightx2v/models/networks/hunyuan_image3/weights/common.py`

- Dense MLP 接入 micro-major gate/up；
- MoE 构建唯一的 phase-neutral FlashInfer pack；
- AR 返回一个 micro view，denoise 返回两个 micro view；
- expert leaf 被最终 pack 替换，防止常驻双份 expert 权重。

### 6.6 动态执行路径

`lightx2v/models/networks/hunyuan_image3/infer/transformer_infer.py`

- attention head 数、TP/SP group 均按当前 phase 动态解析；
- dense/eager MoE 使用 phase-aware linear；
- FlashInfer denoise 执行两个 local micro call 后相加；
- 每个完整 MLP 输出只在活动 TP group 上 all-reduce 一次。

`lightx2v/models/networks/hunyuan_image3/infer/post_infer.py`

- AR lm_head 使用活动 TP4；
- gather 后按 logical order 恢复 vocab shard 顺序。

`lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py`

- 文本生成前激活 AR；
- diffusion 前激活 denoise；
- TP/SP/control group 动态查询；
- 强制当前 TP2+SP2 配置使用 serial CFG；
- AR 与 denoise 选择各自的 FlashInfer cache。

### 6.7 配置、脚本与测试

- 生产配置：`configs/hunyuan_image3/hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.json`
- 1-step smoke 配置：`configs/hunyuan_image3/hunyuan_image3_t2i_phase_parallel_smoke.json`
- 启动脚本：`scripts/hunyuan_image3/run_hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.sh`
- 权重布局测试：`test_cases/test_hunyuan_image3_hybrid_tp_weights.py`
- 阶段执行测试：`test_cases/test_hunyuan_image3_phase_execution.py`
- 4-rank NCCL 拓扑测试：`test_cases/test_hunyuan_image3_phase_topology_distributed.py`

启动脚本还会把 `/opt/conda/bin` 显式加入子进程 `PATH`。FlashInfer 即使命中 autotune 配置，首次 materialize JIT module 时仍可能按名称调用 `ninja`；仅指定 `/opt/conda/bin/torchrun` 并不能保证其子进程能找到同目录下的 `ninja`。

### 6.8 从启动脚本到图片输出的完整调用链

下面的行号对应本报告生成时的代码快照，用于把业务阶段和真实函数一一对上：

| 层次 | 代码入口 | 主要职责 |
| --- | --- | --- |
| 进程启动 | `scripts/hunyuan_image3/run_hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.sh` | 固定 GPU 0–3，启动 4 个 torchrun worker，指定新 JSON |
| 通用 CLI | `lightx2v/infer.py:329-353` | 读取配置、初始化 distributed、创建 runner、调用 `run_pipeline()` |
| 配置归一化 | `lightx2v/utils/set_config.py:46-160` | 解析 phase-aware schema，计算 micro-shard 数并做静态约束检查 |
| 并行组创建 | `lightx2v/utils/set_config.py:604-628` | 构造分阶段 context，并把 denoise mesh 暴露给旧组件 |
| 拓扑实现 | `lightx2v/models/networks/hunyuan_image3/parallel.py:263-375` | 一次创建 TP2/SP2 组，登记 AR WORLD 组和逻辑 rank 映射 |
| 模型存储拓扑 | `lightx2v/models/networks/hunyuan_image3/model.py:182-198` | 将模型级 TP 固定为 storage TP2，而不是当前活动 TP |
| 官方权重读取 | `lightx2v/models/networks/hunyuan_image3/model.py:252-294,343-369` | 判定权重类别，从官方 tensor 直接选 A/B 并放到本卡 |
| T2I 编排 | `lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py:1546-1561` | 依次执行 COT/recaption、denoise、VAE decode |
| AR 入口 | `hunyuan_image3_runner.py:1098-1134` | 激活 AR、生成增强文本、使用 AR 专属 FlashInfer cache |
| denoise 入口 | `hunyuan_image3_runner.py:1345-1410` | 激活 denoise、构造 serial CFG 两分支、进入采样循环 |
| 动态 Transformer | `lightx2v/models/networks/hunyuan_image3/infer/transformer_infer.py:93-120,439-557,649-815` | 动态解析 TP/SP、执行 attention、Dense/MoE 和归约 |
| AR logits | `lightx2v/models/networks/hunyuan_image3/infer/post_infer.py:95-110` | 只投影最后一个 token，并按逻辑 shard 顺序拼接词表 |

按运行顺序展开，调用关系是：

```text
run_hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.sh
└─ torchrun --nproc_per_node=4 -m lightx2v.infer
   └─ infer.main()
      ├─ set_config()
      │  └─ auto_calc_config()
      │     └─ _normalize_hunyuan_image3_phase_parallel()
      ├─ init_parallel_env()
      ├─ set_parallel_config()
      │  └─ build_hunyuan_image3_parallel_context()
      ├─ init_runner()
      │  └─ runner.init_modules()
      │     └─ HunyuanImage3Model._init_tensor_parallel()
      │        └─ 逐个 safe_open(官方 safetensors)
      │           └─ _select_tensor_parallel_shard()
      └─ runner.run_pipeline()
         └─ generate_t2i()
            ├─ _generate_cot_text()
            │  ├─ activate_phase("ar")
            │  └─ AR TP4 forward / token sampling
            ├─ _prepare_text_to_image_inputs()
            ├─ _denoise_latents()
            │  ├─ activate_phase("denoise")
            │  └─ 50 × [cond forward + uncond forward + scheduler step]
            └─ _decode_latents()（仅输出 rank）
```

这个调用链中的关键设计是把“加载拓扑”和“执行拓扑”解耦：`model.py` 只认识固定 storage TP2；算子在每次 `apply()` 时通过 `parallel_context` 查询当前是 TP4 还是 TP2+SP2。

### 6.9 配置如何推导出 ABAB 和 a1b1a2b2

配置归一化首先计算：

```text
storage_tp_size   = 2
ar_tp_size        = 4
denoise_tp_size   = 2
denoise_sp_size   = 2
micro_shard_count = ar_tp_size / storage_tp_size = 2
```

代码强制满足：

```text
storage_tp_size == denoise_tp_size
ar_tp_size % storage_tp_size == 0
micro_shard_count == denoise_sp_size
ar_tp_size == denoise_tp_size * denoise_sp_size == world_size
```

对物理全局 rank `g`，当前 2×2 mesh 下有：

```text
storage_tp_rank = g % denoise_tp_size
local_micro_id  = g // denoise_tp_size
logical_tp_rank = storage_tp_rank * micro_shard_count + local_micro_id
```

代入 `g=0,1,2,3`：

```text
physical rank:      0   1   2   3
storage_tp_rank:    0   1   0   1      -> A B A B
local_micro_id:     0   0   1   1
logical_tp_rank:    0   2   1   3      -> a1 b1 a2 b2
```

这里 `logical_tp_rank` 描述权重在官方连续 TP4 切片中的语义位置；它不改变 WORLD 的物理 rank。`logical_gather_order` 通过对 `physical_to_logical=[0,2,1,3]` 排序得到同样的 `[0,2,1,3]`，专门用于需要有序拼接的输出。

配置检查还验证 Q head、KV head、Dense/MoE intermediate、shared expert intermediate 和 vocab size 都能被最细粒度 TP4 整除。这样错误会在 80B 权重加载前暴露，而不是在某一层 `reshape()` 或 collective 时才失败。

### 6.10 官方权重到常驻权重的逐类切片算法

官方线性层使用常见 checkpoint 形状 `[out_features, in_features]`；LightX2V Default MM 在加载后以 `[in_features, out_features]` 供 `torch.mm(input, weight)` 使用。因此要区分“checkpoint 切片维”和“运行时 view 维”：

| 权重 | 官方 checkpoint 处理 | 本卡常驻 MM 布局 | AR view | denoise view |
| --- | --- | --- | --- | --- |
| 普通 column / `lm_head` | 沿 dim 0 选 A/B | `[in, out/2]` | 沿 dim 1 `narrow` 为 `[in, out/4]` | 完整 `[in, out/2]` |
| row / `o_proj` / `down_proj` | 沿 dim 1 选 A/B | `[in/2, out]` | 沿 dim 0 `narrow` 为 `[in/4, out]` | 完整 `[in/2, out]` |
| grouped QKV | 沿完整 KV group 选 A/B | `[hidden, local_qkv]` | 再按完整 KV group 选 1/2 resident view | 完整 resident QKV |
| fused gate/up | 官方 `[gate_all, up_all]` 重组并选 A/B | `[hidden, g0,u0,g1,u1]` | 一个连续 `[hidden, gi,ui]` view | 完整 micro-major view |
| norm、router、embedding 等 | 不属于 `_tp_split_type()` | 每卡复制 | 复制 | 复制 |

实际加载伪代码与 `model.py::_load_safetensor_to_dict()` 一致：

```python
for official_file in checkpoint_files:
    with safe_open(official_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)                  # 官方 tensor
            tensor = select_storage_shard(key, tensor) # 只选 TP2 的 A 或 B
            weight_dict[key] = tensor.to(local_gpu, dtype)
```

这个过程中没有先生成一个完整的 80B GPU tensor，也没有 rank 间发送参数。四个 rank 各自扫描官方文件；每个 key 先以完整 CPU tensor 物化，完成本地切片后只把需要的 TP2 shard 放入本卡。其直接代价是启动 I/O、CPU tensor 读取和 host memory 瞬时占用重复，收益是无需维护离线格式和版本对应关系。

运行时 `HunyuanImage3HybridTensorParallelLinear.active_weight` 的逻辑是：

```python
if active_tp_size == storage_tp_size:   # denoise TP2
    return resident_weight              # 完整 A 或 B
if active_tp_size == storage_tp_size * micro_shard_count:  # AR TP4
    return resident_weight.narrow(..., local_micro_id * width, width)
```

`Tensor.narrow()` 返回共享底层 storage 的 view。它不执行 `.contiguous()`、`.clone()`、`.to()` 或 collective；测试也检查了 resident tensor 与 AR view 的 storage 共享关系。只有官方权重刚读入时的 TP2 选择和 gate/up/FlashInfer 一次性组织允许发生 copy。

#### grouped QKV 为什么不能直接四等分

QKV 的官方语义布局为：

```text
[num_kv_heads, q_heads_per_kv + 2, head_dim]
```

其中 `+2` 是同一 KV group 的 key 和 value。`select_grouped_qkv_storage_shard()` 先验证：

```text
num_attention_heads % num_key_value_heads == 0
num_key_value_heads % (storage_tp_size * micro_shard_count) == 0
```

然后以 KV group 为单位选取连续区间。AR 的 `_micro_view()` 也按 `qkv_group_width=(q_per_kv+2)*head_dim` 计算起点和宽度，保证 Q/K/V 的语义配对不会被切断。

#### fused gate/up 如何同时服务两个阶段

对 intermediate 维记为 `I`，官方权重为 `[2I,H]`：

```text
官方:       [gate_0 ... gate_3 | up_0 ... up_3]
rank A:     [gate_0 | up_0 | gate_1 | up_1]
rank B:     [gate_2 | up_2 | gate_3 | up_3]
```

AR 时每个 rank 的 view 天然是 `[gate_i,up_i]`。denoise 时不恢复权重顺序，而把 projection 输出 reshape 为 `[token,micro,2,width]`，逐 micro 执行代码中既有的 `gate * silu(up)`，再 flatten 为 `[micro0_active,micro1_active]`。该顺序与 `down_proj` 的 resident 输入行一致，因此无需搬回 `[gate_all,up_all]`。

### 6.11 TP collective 的算子等价性

对一个 row-parallel 算子，把规范 TP4 权重记为 `W0,W1,W2,W3`，对应 `a1,a2,b1,b2`。AR 中每个物理 rank 选择自己的 micro 权重，WORLD all-reduce 得到：

```text
Y_AR = X0·W0 + X1·W1 + X2·W2 + X3·W3
```

物理执行顺序虽然是 `W0,W2,W1,W3`，但 all-reduce 是求和，交换项次序不改变数学结果。

denoise 中，A resident 等价于 `W0+W1` 对应的完整输入分片，B resident 等价于 `W2+W3`。在每个 SP lane 内，TP 组 `[0,1]` 或 `[2,3]` 计算：

```text
Y_denoise = X_A·[W0,W1] + X_B·[W2,W3]
          = X0·W0 + X1·W1 + X2·W2 + X3·W3
```

`HunyuanImage3HybridTensorParallelLinear.apply()` 在 row/down/o projection 后查询 `active_tp_group`：AR 选择 WORLD，denoise 选择当前 TP2 组；bias 在 reduction 之后只加一次。

column-parallel 输出通常不需要归约，但消费者必须沿用同一 shard 语义。唯一需要全局有序结果的核心路径是 AR `lm_head`：四个 rank 先在 WORLD `all_gather`，再按 `[0,2,1,3]` 拼接，恢复 `[vocab_q0,vocab_q1,vocab_q2,vocab_q3]`。代码只对最后一个 token 做 vocab projection 和 gather，避免复制整个 prompt 的 logits。

### 6.12 FlashInfer MoE 的单套通用 pack

普通 `narrow()` view 不能直接满足 FlashInfer 对 expert pack 连续性的要求，所以 `ensure_flashinfer_weights()` 在首次 AR 使用时构建一次最终常驻 pack：

```text
W1 = [micro=2, expert=64, 2*intermediate_micro, hidden]
W2 = [micro=2, expert=64, hidden, intermediate_micro]
```

构建采用“分配最终 pack -> 逐 expert copy -> 立刻清空原 expert leaf”的方式，避免稳态保留原 expert tensor 和最终 pack 两份常驻，并限制构建期间新旧权重的瞬时重叠。初始化完成后，再请求不同 device/dtype 会直接报错，防止运行时静默复制第二套权重。

活动 shard 选择为：

```text
AR / active_tp_size=4:
    每卡只返回 local_micro_id 对应的一个连续 view

denoise / active_tp_size=2:
    每卡返回 resident A 或 B 内的两个连续 micro view
```

FlashInfer 始终收到逻辑 TP4 元数据。对单个 token/expert routing 结果，将四个 micro partial 记为 `P0..P3`：

```text
AR:
    rank 0 -> P0 (logical rank 0)
    rank 1 -> P2 (logical rank 2)
    rank 2 -> P1 (logical rank 1)
    rank 3 -> P3 (logical rank 3)
    WORLD all-reduce -> P0+P1+P2+P3

denoise:
    A rank local: P0 + P1
    B rank local: P2 + P3
    TP2 all-reduce -> P0+P1+P2+P3
```

`_infer_mlp_flashinfer()` 对两个 local micro 使用相同的 router top-k 结果，逐次调用 kernel，并通过 `combined_output.add_()` 本地相加；外层 `infer_mlp()` 最后只执行一次活动 TP all-reduce。这样不会把一次逻辑 MLP 错误地归约两次。BF16 加法次序与一次 full-local kernel 不完全相同，所以保证的是数值近似等价，不是 bitwise 等价。

### 6.13 denoise 中 TP2 与 Ulysses SP2 如何配合

denoise 激活后，模型动态看到 `active_tp_size=2` 和 `active_seq_size=2`：

1. `_seq_parallel_pre_process()` 将全序列补齐到 2 的倍数，按 SP rank 切出本地连续一半，并记录原长度、padding、起点和有效长度。
2. QKV projection 按 TP2 产生本地 head；`infer_attention()` 用全局 head 数除以当前活动 TP2，而不是沿用初始化时的常量。
3. Ulysses 在 `[0,2]` 或 `[1,3]` 上执行 `all2all_seq2head()`，将“半序列、较多 head”交换成“完整序列、较少 head”。这两个组内的 rank 持有同一个 A 或 B storage shard。
4. attention 完成后 `all2all_head2seq()` 恢复本 rank 的序列片段。
5. `o_proj` 和 MLP down projection 在 `[0,1]` 或 `[2,3]` 的 TP2 组上 all-reduce。
6. Transformer 尾部 `_seq_parallel_post_process()` 在 SP 组内 gather 序列并去掉 padding；buffer key 包含 phase 和 group identity，防止不同拓扑复用错误缓存。

SP2 只切激活，不切新的权重。正是因为 rank 0/2 都是 A、rank 1/3 都是 B，Ulysses 跨 SP rank 交换激活时，权重语义仍保持一致。

### 6.14 phase 切换的同步协议

`activate_phase()` 的核心协议可简化为：

```python
all_gather_world(local_current_phase, requested_target_phase)
assert all_ranks_have_same_current_and_target()
if target != current:
    torch.cuda.synchronize()
    dist.barrier(WORLD)
    self._phase = target
```

即使目标 phase 与当前 phase 相同，也会执行第一步 consensus。原因是如果某个 rank 认为这是 no-op 并提前返回，而另一个 rank 认为需要 barrier，两者会进入不同 collective 序列并挂起。只有 consensus 通过后才允许 no-op 返回。

切换完成后，所有动态属性从预先登记的 `_PhaseParallelState` 读取。此函数不调用 `init_process_group()`、`new_group()`、`to()`、`copy_()` 或 checkpoint loader。日志中的 allocated/reserved 快照用来辅助确认阶段边界没有出现一套新模型权重。

### 6.15 runner、CFG 和 autotune 的业务控制

runner 不把 AR/denoise 拆成两个模型实例，而是在同一个 `generate_t2i()` 内显式设置阶段：

- `_generate_cot_text()` 首行激活 `ar`，随后 think/recaption 的 prefill 与逐 token decode 都使用 WORLD TP4；
- `_denoise_latents()` 首行激活 `denoise`，后续所有 step 都使用 TP2+SP2；
- 混合 TP/SP 时 `_parallel_control_group()` 返回 WORLD，使随机 latent、采样 token 和 runner 状态在四卡一致；
- `flashinfer_autotune_cache_by_phase` 依据 context 当前 phase 选择不同 JSON，避免两阶段调优结果相互覆盖并便于独立复现；代码不依赖跨 phase 复用是否可行；
- serial CFG 每个 step 先执行 conditional，再执行 unconditional，然后计算 `uncond + scale * (cond-uncond)`。每次 Transformer forward 的 batch 都为 1，符合当前 Ulysses 路径约束；
- denoise 完成后非 rank 0 返回空结果，只有 rank 0 进入 VAE decode 和 PNG 保存。VAE 加载日志因此不代表 Transformer 权重被重载。

#### Collective 时序总表

| 代码位置 | Collective | AR 使用的组 | denoise 使用的组 | 作用 |
| --- | --- | --- | --- | --- |
| `set_config.py:659-665` | `all_reduce` | WORLD | WORLD | 初始化后预热 distributed |
| `parallel.py:88-123` | `all_gather_into_tensor` | WORLD | WORLD | 核对所有 rank 的 current/target phase |
| `parallel.py:142-149` | CUDA sync + `barrier` | WORLD | WORLD | 真正 phase 变化时清空上一阶段排队工作 |
| `hybrid_tp.py:344-349` | `all_reduce` | WORLD TP4 | TP2 `[0,1]` / `[2,3]` | row-linear partial sum |
| `transformer_infer.py:677-682` | `all_reduce` | WORLD TP4 | 同上 TP2 | MoE 输出的最终 partial sum |
| `post_infer.py:102-109` | `all_gather` | WORLD TP4 | diffusion 分支不进入 | 收集并重排 vocab logits |
| `hunyuan_image3_runner.py:512-518` | `broadcast` | WORLD | WORLD | 同步 sampled token、初始 latent、prediction |
| `all2all.py` 的 Ulysses helper | `all_to_all_single` | - | SP2 `[0,2]` / `[1,3]` | seq-to-head 与 head-to-seq |
| `model.py:531-547` | `all_gather_into_tensor` | 不进入 | 同上 SP2 | Transformer 尾部恢复完整序列 |
| `hunyuan_image3_runner.py:520-532,1498` | `barrier` | WORLD | WORLD | runner/denoise 收尾同步 |

本方案 `cfg_p_size=1` 且使用 serial CFG，因此旧 CFG-parallel 路径中的 cond/uncond `all_gather` 不会触发。

### 6.16 原始入口兼容与文件隔离

本次没有覆盖用户给出的原始文件：

| 用途 | SH | JSON | GPU/并行语义 |
| --- | --- | --- | --- |
| 原始任务，保持不变 | `run_hunyuan_image3_t2i_tp_sp_cfg_flashinfer.sh` | `hunyuan_image3_t2i_tp_sp_cfg_flashinfer.json` | 8 卡，TP2+SP2+CFG2 |
| 新分阶段任务 | `run_hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.sh` | `hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.json` | 4 卡，AR TP4 / denoise TP2+SP2 / serial CFG |

代码层也采用显式开关隔离：只有 `model_cls == "hunyuan_image3"` 且 `parallel.phase_aware == true` 才构建新 context。phase-aware 配置把 denoise 的 `tensor_p_size=2`、`seq_p_size=2` 写回 flat compatibility alias，尚未改造成动态查询的旧组件仍看到合法的 denoise mesh；AR-aware 算子则只从 context 读取活动组。没有 `phase_aware` 的旧配置继续走原 `set_parallel_config()` 分支。

## 7. 配置与运行

生产配置的核心部分为：

```json
{
  "parallel": {
    "pipeline_parallel": false,
    "phase_aware": true,
    "storage_tensor_p_size": 2,
    "ar": {
      "tensor_p_size": 4,
      "seq_p_size": 1
    },
    "denoise": {
      "tensor_p_size": 2,
      "seq_p_size": 2
    },
    "cfg_p_size": 1,
    "seq_p_attn_type": "ulysses",
    "cfg_mode": "serial"
  },
  "flashinfer_autotune_cache_by_phase": {
    "ar": "save_results/hunyuan_image3_flashinfer_autotune_t2i_ar_tp4.json",
    "denoise": "save_results/hunyuan_image3_flashinfer_autotune_t2i_denoise_tp2_sp2.json"
  }
}
```

直接运行：

```bash
cd /data/liuhongda/LightX2V
bash scripts/hunyuan_image3/run_hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.sh
```

脚本固定：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
/opt/conda/bin/torchrun --standalone --nproc_per_node=4 ...
```

可以用 `TORCHRUN_BIN` 环境变量覆盖 torchrun 路径，但不需要也不允许传入转换后的 checkpoint。

## 8. 测试与实验结果

### 8.1 静态与聚焦测试

| 验证项 | 结果 |
| --- | --- |
| Ruff changed-file lint | 通过 |
| Python `py_compile` | 通过 |
| JSON 解析 | 通过 |
| shell `bash -n` | 通过 |
| `git diff --check` | 通过 |
| gate/up micro-major + view 共享 | 通过 |
| grouped QKV 完整 KV group 切分 | 通过 |
| row/down micro view | 通过 |
| FlashInfer pack 连续性与 storage 共享 | 通过 |
| runner phase、CFG、cache、logits、head、FI 执行测试 | 6/6 通过 |
| 4-rank NCCL 实际 topology/collective | 通过 |
| 4-rank phase 状态分歧注入 | 所有 rank 同步报错，无 barrier 挂起 |
| 4-rank 非法 phase target 注入 | 所有 rank 同步报错，无 collective 错序 |

环境中没有安装 pytest，因此不依赖 pytest fixture 的测试函数由 Python 直接调用，`unittest` 用标准 runner 执行。4-rank 测试使用真实 NCCL 和 GPU 0–3，不是 mock。

NCCL topology 测试输出：

```json
{
  "ar_tp": [0, 1, 2, 3],
  "denoise_sp": [[0, 2], [1, 3]],
  "denoise_tp": [[0, 1], [2, 3]],
  "physical_to_logical": [0, 2, 1, 3],
  "status": "ok"
}
```

### 8.2 FlashInfer 单算子实验

在 GPU 4 上额外进行了单卡 FlashInfer MoE 实验，用于隔离“denoise 两次 micro call”本身的开销。真实尺寸配置为：

- 64 experts；
- hidden size 4096；
- TP2-local intermediate 1536；
- 每个 micro intermediate 768；
- 1024 tokens；
- top-k 2；
- BF16。

| 路径 | 平均耗时 |
| --- | ---: |
| 完整 TP2-local 一次调用 | 3.5506 ms |
| 两次 micro 调用并本地相加 | 3.6008 ms |
| 相对开销 | +1.42% |

数值对比：

- max absolute error：0.0029297；
- mean absolute error：0.0001481；
- cosine similarity：0.9999929。

误差来自 BF16 下不同的加法/累加顺序。较小尺寸实验中，两份 quarter sum 与 full-local 的 cosine similarity 为 0.999993。另一个控制实验确认，相同权重只改变 FlashInfer 的 `tp_size/tp_rank` 元数据时结果 bitwise 相同。

以上是单个 MoE operator 的实验，不能直接当作端到端加速比；它用于证明两次 micro call 的额外 kernel 开销较小且数值一致性可接受。

### 8.3 1024×1024、1-step smoke

使用 smoke 配置完成了官方权重加载、AR、phase switch、1-step denoise、VAE decode 和图片保存，进程正常退出。

曾尝试将 smoke 分辨率降到 512×512，但官方模型的 image-token geometry 仍要求 4096 image tokens，而 512 配置只产生 1024 个 patch token，pre-infer scatter 因形状不匹配失败。该问题与并行实现无关，因此 smoke 配置恢复到官方 1024×1024 几何。

### 8.4 首次完整 50-step 生产运行

首次运行在同一次任务中先后创建 AR 和 denoise 两套 FlashInfer cache，因此属于 cold/autotune 数据。

| 阶段 | 耗时 | Rank 0 总耗时占比 |
| --- | ---: | ---: |
| Transformer 官方权重读取与初始化 | 234.334 s | 60.11% |
| AR 前准备 | 2.209 s | 0.57% |
| AR TP4（含首次 autotune） | 92.739 s | 23.79% |
| denoise TP2+SP2，50 step（含首次 autotune） | 46.482 s | 11.92% |
| VAE 加载、解码与保存 | 13.641 s | 3.50% |
| GC/收尾 | 0.420 s | 0.11% |
| Rank 0 profile 总计 | 389.825778 s | 100% |

denoise 首步包含调优，耗时 29.08 s；其余 49 步约 17.40 s，稳定后约 0.355 s/step，即 2.82 step/s。AR 和 denoise 各生成 18 条 FlashInfer 配置，证明两套 cache 已按 phase 隔离。

各 rank profile：

| Rank | Profile 时间 |
| --- | ---: |
| 0 | 389.825778 s |
| 1 | 376.045685 s |
| 2 | 376.060188 s |
| 3 | 376.013813 s |

rank 1–3 的最大差异只有 46.375 ms。denoise 收尾 barrier 位于 VAE 之前；随后非输出 rank 直接返回并跳过 VAE，所以它们的 profile 区间比 rank 0 提前约 13.8 s 结束。

### 8.5 Cache 命中的完整 50-step 复跑

第二次完整运行显式保留 `/opt/conda/bin` 在子进程 `PATH` 中，并复用首次生成的两套 cache。四个 rank 均记录：

- AR：`effective_tune_mode=False`，加载 18 条配置，gemm1/gemm2 均为 config-file cache hit；
- denoise：`effective_tune_mode=False`，加载 18 条配置；
- 每个 rank 仍直接读取官方 32 个 checkpoint 文件；
- 无 traceback、OOM、NCCL 错误或 NaN；
- 最终图片保存成功，四个 rank 正常清理进程组。

Rank 0 时间拆分：

| 阶段 | Warm-cache 实测 |
| --- | ---: |
| Transformer 官方权重读取与初始化 | 240.667 s |
| AR 前准备 | 2.201 s |
| AR TP4 | 31.760 s |
| denoise TP2+SP2，50 step | 16.980 s |
| VAE 加载、解码与保存 | 14.580 s |
| GC/收尾 | 0.257 s |
| Rank 0 profile 总计 | 306.445821 s |

denoise 的 50 步总体平均约 2.95 step/s；首步包含 JIT module materialize，耗时 1.87 s，后续稳定在约 3.25 step/s。各 rank profile 为：

| Rank | Warm-cache profile |
| --- | ---: |
| 0 | 306.445821 s |
| 1 | 291.897873 s |
| 2 | 291.894152 s |
| 3 | 291.872830 s |

相对首次含调优运行的 389.825778 s，第二次观测到端到端减少 83.379957 s，即 21.39%。其中 denoise 从 46.482 s 降到 16.980 s。AR 文本启用了 sampling，两次 think/recaption 的内容和长度不同，因此 AR 92.739 s 到 31.760 s 的变化不能全部归因于 cache；它应视为本次业务实测，而不是严格的同 token benchmark。

该次 warm 运行当时记录的输出为 1024×1024 RGB PNG，SHA-256 为 `ec2b102f9c8cbb962ddf66c4801e92b8b1bb13968a21e9a070d419f4da3a6ab3`。人工检查同样通过：高速公路、手机通话的驾驶员和副驾驶金毛均清晰可见。该历史 warm 文件当前未保留在工作区，不能用当前文件重新校验此哈希。

第一次手写 warm 命令曾只使用绝对路径启动 torchrun，却没有让 `/opt/conda/bin` 进入 child-process `PATH`。cache 成功读取后，FlashInfer JIT 因找不到 `ninja` 退出。该失败用于定位部署环境问题；目标启动脚本已增加显式 `PATH`，随后上述复跑成功。它不涉及模型并行或权重逻辑错误。

### 8.6 阶段边界显存

四张卡日志记录完全一致：

| 边界 | allocated / 每卡 | reserved / 每卡 |
| --- | ---: | ---: |
| AR 激活 | 76.564 GiB | 77.395 GiB |
| denoise 激活 | 77.936 GiB | 82.436 GiB |
| 变化 | +1.372 GiB | +5.041 GiB |

这些值是 phase 激活边界的快照，不是阶段内 `max_memory_allocated`。reserved 的增长包含 CUDA allocator cache，不能等同于活跃 tensor 增长。但变化量远小于再常驻一套 80B TP 权重所需的显存，且四个 rank 在 denoise 前没有任何 checkpoint 读取，支持“单套权重常驻、阶段切换零搬运”的实现结论。

warm-cache 复跑的四卡边界值同样完全一致：AR 为 76.564 GiB allocated / 77.395 GiB reserved，denoise 为 76.718 GiB allocated / 81.225 GiB reserved。相比首次调优运行，denoise 边界显存更低；该差异可能与调优/JIT 的临时 workspace 或 allocator 状态有关，但当前日志没有 allocation attribution，不能据此确定唯一原因。

### 8.7 阶段切换与单次加载证据

四个 rank 均满足：

- AR 前各出现且仅出现 32 条 Transformer safetensors 加载记录；
- AR 激活后 Transformer 加载记录为 0；
- 四个 rank 的 AR/denoise 激活时间对齐到 1 ms 以内；
- COT 完成日志为 11:37:56.795；
- denoise 激活日志为 11:37:56.810；
- 两条日志间隔约 15 ms；它包含 runner 调用、phase consensus、CUDA synchronize、barrier 和日志开销，不是独立 CUDA-event 测得的纯切换延迟；
- 无 CUDA OOM、NCCL timeout、collective failure、NaN 或 traceback；
- 四个 rank 均正常 destroy distributed process group。

额外的故障注入覆盖 phase 状态分歧和单-rank非法 target：`activate_phase()` 即使在本 rank 看来是 no-op，也会先在 WORLD 上核对 current/target phase。进程被 SIGKILL、GPU hard fault，或单个 rank 在模型 TP/SP collective 中途抛出异常时，仍需要 torch elastic/NCCL timeout 负责终止；不能在未知 collective 序号上安全插入另一个 Python 层状态 collective。

### 8.8 最终输出

启动脚本会复用同一个输出路径，所以后续实验会覆盖之前的 PNG。当前工作区可验证的最新文件为：

```text
/data/liuhongda/LightX2V/save_results/
  hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer.png
```

当前文件属性：

- 1024×1024；
- RGB PNG；
- 1,267,766 bytes；
- 修改时间：2026-08-07 02:08:38 UTC；
- SHA-256：`9b1f48aebe43dfa61886a90cd4eace42920237dee495144b9a118b24d56d77fe`。

人工复查当前文件：画面明确包含高速公路、驾驶员手持电话和副驾驶拉布拉多犬，空间关系、人物手部和车辆环境均可辨认，说明 AR、denoise、CFG、VAE 全链路正常。

8 月 6 日 cold/warm 运行当时记录过不同哈希，其中 warm 文件曾使用：

```text
/data/liuhongda/LightX2V/save_results/
  hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_flashinfer_warm.png
```

该历史文件当前没有保留，同名生产 PNG也已被 8 月 7 日任务覆盖，因此历史哈希只作为运行记录，不能与当前文件混为同一产物。后续正式 benchmark 应在文件名中加入日期或 run-id，避免覆盖证据。

### 8.9 权重初始化后的 AR + denoise 专项计时

针对“权重加载之后，AR+denoise 总共需要多少秒”，2026-08-07 又执行了一次 4 卡、50-step、1024×1024、FlashInfer cache 命中的完整 T2I。计时通过 rank 0 日志时间戳人工拆分，边界如下：

| 边界 | UTC 时间 | 与上一边界间隔 |
| --- | --- | ---: |
| Transformer/runner 初始化完成 | 01:52:39.555 | - |
| AR phase 激活日志 | 01:52:42.191 | 2.636 s（AR 前业务准备） |
| COT/recaption 完成日志 | 01:53:16.884 | 34.693 s（AR） |
| denoise phase 激活日志 | 01:53:16.896 | 0.012 s（阶段边界观测间隔） |
| VAE 权重加载开始 | 01:53:33.785 | 16.889 s（50-step denoise） |

因此有两个有用口径：

```text
纯 AR + 阶段边界 + denoise
= 34.693 + 0.012 + 16.889
= 51.594 s

从 Transformer/runner 初始化完成到 denoise 结束
= 2.636 + 51.594
= 54.230 s
```

对业务问题最直接的回答是：**权重初始化完成后，到 AR 和 denoise 全部结束为 54.230 秒；若只统计 AR 激活后的两段核心计算及其边界，则为 51.594 秒。** 两者都排除了 VAE 加载、解码与保存。

本次 denoise 平均约 2.96 step/s，稳定段约 3.25 step/s。rank profile 总时间为：

| Rank | 2026-08-07 profile |
| --- | ---: |
| 0 | 483.397703 s |
| 1 | 464.857815 s |
| 2 | 464.771302 s |
| 3 | 464.827019 s |

完整 profile 包含官方权重读取/初始化、AR、denoise，以及 rank 0 的 VAE/保存。按现有日志粗拆，初始化前段约 410.331 s，已从上述 54.230 s 中排除。这个 410.331 s 不能解释为“纯磁盘读取耗时”：基础 loader 的 `Loading weights` 日志写在真正读取该文件之前，当前没有在 `_load_ckpt()` 返回和 `_apply_weights()` 完成处设置专用 CUDA/wall timer。

两次 cache 命中运行的重复性如下：

| 运行 | 纯 AR+边界+denoise | 初始化完成到 denoise 结束 |
| --- | ---: | ---: |
| 2026-08-06 warm | 48.740 s | 50.941 s |
| 2026-08-07 专项复测 | 51.594 s | 54.230 s |
| 两次算术平均 | 50.167 s | 52.586 s |

denoise 两次分别约 16.980 s 和 16.889 s，较稳定；差异主要来自 AR。当前配置 `text_do_sample=true`，think/recaption 的实际生成内容和长度会影响 AR wall time，所以 50.167 s 是本 prompt、seed、软件环境下的两次业务观测平均值，不是固定 token 数 benchmark。

计时边界还有三点必须说明：

1. `COT 完成 -> denoise 激活` 的 12 ms 包含 Python 调用、phase consensus、CUDA synchronize、WORLD barrier 和日志开销，不是 `activate_phase()` 的纯 kernel 延迟。
2. 以“VAE 权重加载开始”作为 denoise 结束标志，是因为代码在 denoise 最终 barrier 之后才让 rank 0 进入 VAE；VAE 时间没有混入 16.889 s。
3. 仓库当前没有专用 benchmark 脚本或原始日志 JSON，这组数字是一次人工日志拆分。若作为长期性能门禁，应在代码中增加带 `torch.cuda.synchronize()` 的结构化 phase timer，并以 run-id 保存原始结果。

## 9. 性能与业务分析

### 9.1 已消除的开销

如果 AR TP4 和 denoise TP2+SP2 分别维护独立权重布局，阶段切换通常需要重新加载 checkpoint、跨卡重分布，或同时常驻两套切分。80B 模型下，这三种方式分别带来数分钟 I/O、显著 NVLink 流量，或不可接受的额外显存。

当前实现把阶段差异降为：

- 选择已存在的 process group；
- 选择同一 storage 上的完整 view 或 micro view；
- 必要时对输出激活做逻辑顺序恢复。

因此切换本身不再与模型参数规模线性相关。

### 9.2 当前主要瓶颈

首次完整运行中，60.11% 时间用于官方权重读取和初始化。这是坚持“每个 rank 直接读取官方权重、无离线转换”的直接代价。当前四个进程都会扫描全部 32 个文件，并逐 key 在 CPU 侧取得完整 tensor；对于 TP 权重，随后只把本 rank 需要的 A/B slice 常驻到 GPU。

AR 的具体 token 数随 think/recaption 内容变化，日志中没有稳定的 tokens/s 指标，因此不应从单次 wall time 推导通用 AR 吞吐。

denoise 使用 serial CFG，每个 step 顺序执行 conditional 和 unconditional 两个 forward。其 2.82 step/s 已包含这一业务逻辑。

### 9.3 FlashInfer 两次 micro call

denoise 的 A/B 各包含两个 TP4 micro-shard。为了同时复用 AR 的 contiguous pack 且不移动权重，denoise 每个 expert MLP 做两次 FlashInfer 调用。单算子实验显示相对一次 full-local 调用约增加 1.42%，这是“通用布局、零切换搬运”换来的小幅 kernel launch 成本。

### 9.4 数值边界

micro partial output 的本地求和改变了 BF16 累加顺序，因此不保证 bitwise 等价。单算子 cosine similarity 为 0.9999929，完整生成也成功，但当前实验没有与官方单卡或旧 TP 路径做逐层 FP32 golden comparison。若用于严格回归门禁，应额外保存固定 prompt/seed 的关键层统计量和图像相似度阈值。

## 10. 已知限制

当前经过实现与实机验证、建议正式使用的 phase-aware 范围是：

- 本次配置为单机 world size 4；构建器本身按“两个 phase 都覆盖 world”做通用检查，并未硬编码只能为 4；
- 官方未量化 Default MM 权重；
- AR TP4，denoise TP2+SP2；
- `cfg_p_size=1` 和 serial CFG；
- 不支持 LoRA、runtime diff weight；
- 不支持 CPU offload 和 lazy load；
- 不支持 pipeline parallel；
- FlashInfer 只支持当前的 SiLU/SwiGLU expert；
- FlashInfer JIT 运行时需要 `ninja` 可从子进程 `PATH` 找到；目标脚本已显式加入 `/opt/conda/bin`；
- 单-rank若在 FlashInfer context 内的模型 collective 中途异常，其他 rank 依赖 torch elastic/NCCL timeout 退出；
- phase 边界日志不是 CUDA 峰值显存统计；
- 首次启动仍受四 rank 并行读取官方 checkpoint 的 I/O 限制。

旧的非 phase-aware Hunyuan Image 3 配置仍走原分支；新逻辑只在 `parallel.phase_aware=true` 时启用。

## 11. 后续优化建议

按收益和风险排序：

1. 增加 `max_memory_allocated/reserved` 分阶段采样，得到严格峰值而不是边界快照。
2. 给 AR 增加 generated-token 数和 decode tokens/s 指标，区分 prompt prefill、autotune 和逐 token decode。
3. 在不引入离线 checkpoint 的前提下，研究同 storage shard 的两个 SP rank 由一个 rank 读取官方 tensor 后在组内广播，降低重复磁盘 I/O；这会改变启动通信路径，需要单独评估。
4. 为固定 prompt/seed 建立旧实现与新实现的逐层数值和最终图像回归。
5. 如果 FlashInfer 后续支持一次调用多个本地 TP partition，可合并 denoise 的两次 micro call。
6. 若业务允许增加 GPU 数，再评估 CFG parallel，避免 serial CFG 的双 forward。

## 12. 验收清单

- [x] 官方 checkpoint 直接读取，无离线转换工具；
- [x] 只选择 4 张 H200；
- [x] AR 使用 TP4；
- [x] denoise 使用 TP2+SP2；
- [x] 常驻 `ABAB`；
- [x] AR 物理 view 为 `a1b1a2b2`；
- [x] denoise 恢复完整 `ABAB` view；
- [x] 阶段切换无 Transformer reload/跨卡权重搬运；
- [x] 独立 AR/denoise FlashInfer cache；
- [x] 4-rank NCCL 拓扑测试通过；
- [x] 1024×1024、50-step T2I 完整运行成功；
- [x] 输出图片人工检查通过。
