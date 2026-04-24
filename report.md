# Pixelcraft-Diffusion 项目阶段性报告

## 1. 项目概述

本项目当前处于第一阶段：**完成项目初始化，并搭建一个可运行的低分辨率条件扩散模型基础框架**。

项目目标是围绕以下任务展开：

- 输入类别标签或简短文本条件；
- 生成 `32x32` 或 `64x64` 的低分辨率图像；
- 主要面向 Pixel Art、Emoji、Icon 等风格化小图；
- 保留扩散模型的核心流程，包括数据处理、噪声注入、条件建模、训练、采样和评估。

在这一阶段，重点不是追求最终生成质量，而是优先把研发链路打通，确保后续可以在此基础上继续扩展数据集、模型结构和实验方案。

## 2. 当前已完成的工作总览

目前已经完成以下核心工作：

1. 完成服务器连接和仓库初始化梳理。
2. 更新项目 `README.md`，补充中文项目说明和开发流程。
3. 配置 GitHub 推送能力，确保服务器可以直接向远端仓库提交代码。
4. 创建独立的 Conda 环境，避免污染系统 Python 和已有环境。
5. 安装项目所需的基础依赖，包括 PyTorch、torchvision、Pillow、PyYAML、tqdm 和 numpy。
6. 搭建完整的项目目录结构。
7. 实现最小可运行的数据预处理脚本。
8. 实现最小可运行的条件扩散模型训练脚本。
9. 实现采样脚本，可以从 checkpoint 生成图像。
10. 实现简单评估脚本，用于 sanity check。
11. 完成一次端到端 smoke test，验证数据处理、训练、采样、评估链路全部可执行。
12. 将上述工作提交并推送到 GitHub 主分支。

## 3. 服务器与仓库相关工作

### 3.1 服务器连接

已确认当前使用的服务器连接方式为：

```bash
ssh root@124.70.208.92
```

项目仓库路径为：

```bash
/data/anqili/Pixelcraft-Diffusion
```

### 3.2 GitHub 远端配置

原始远端为 HTTPS 地址，但服务器侧无法在非交互环境中输入用户名和 Token，因此后续将仓库 remote 切换为 SSH：

```bash
git@github.com:starrywiki/Pixelcraft-Diffusion.git
```

在你将服务器公钥添加到 GitHub 后，服务器已经可以正常完成：

- Git 提交
- Git 推送
- 与 `origin/main` 同步

### 3.3 已完成的提交

目前至少已经完成并推送以下关键提交：

```text
7a179e0 docs: add project framework README
ca758ec feat: add runnable diffusion framework
```

这些提交已经进入 GitHub 仓库 `main` 分支。

## 4. 文档建设工作

### 4.1 README 重构

已将仓库原本非常简短的 README 扩展为一份中文项目说明文档，内容包括：

- 项目背景与目标；
- 任务定义；
- 推荐数据集；
- 推荐技术路线；
- 数据准备流程；
- 模型设计建议；
- 训练流程；
- 推理与展示方式；
- 评估方式；
- 建议仓库结构；
- GitHub 协作规范；
- 开发里程碑；
- 快速开始说明。

### 4.2 README 的作用

当前 README 已经能够承担以下作用：

1. 让项目合作者快速理解项目目标。
2. 为后续代码组织提供统一结构参考。
3. 指导新成员完成环境配置和基础使用。
4. 作为课程项目或研究项目初始阶段的公开说明。

## 5. 环境配置工作

### 5.1 创建独立环境

已在服务器上创建新的 Conda 环境：

```bash
/data/anqili/conda_env/pixelcraft
```

该环境与系统 Python 隔离，便于后续单独维护项目依赖。

激活方式：

```bash
conda activate /data/anqili/conda_env/pixelcraft
```

### 5.2 服务器 GPU 情况

已检查服务器 GPU，当前可见多张 GPU，型号为：

```text
Tesla V100S-PCIE-32GB
```

这意味着当前服务器完全具备训练低分辨率扩散模型的硬件条件，而且显存余量足够支持后续较大的 batch size 或更复杂模型。

### 5.3 依赖安装情况

已在新环境中安装并验证以下基础依赖：

- `torch`
- `torchvision`
- `Pillow`
- `PyYAML`
- `tqdm`
- `numpy`

其中 PyTorch 已安装为 CUDA 版本，验证结果如下：

```text
torch 2.5.1+cu121
cuda_available True
cuda_devices 8
```

说明：

- 当前环境可以使用 GPU；
- 当前 PyTorch 安装可正常导入；
- 服务器上可见 8 张 GPU；
- 当前项目至少已经可以在单卡上顺利运行 baseline。

## 6. 当前仓库结构

目前仓库已经具备以下基础结构：

```text
Pixelcraft-Diffusion/
  README.md
  report.md
  .gitignore
  requirements.txt
  configs/
    pixelart_32.yaml
    emoji_32.yaml
  data/
    README.md
  scripts/
    prepare_data.py
    train.py
    sample.py
    evaluate.py
  src/
    pixelcraft/
      __init__.py
      data/
        __init__.py
        dataset.py
        transforms.py
      models/
        __init__.py
        conditioning.py
        diffusion.py
        unet.py
      training/
        __init__.py
        trainer.py
      utils/
        __init__.py
        config.py
        image.py
        seed.py
  outputs/
    .gitkeep
```

这套结构已经满足一个标准研究原型项目的基础要求：

- 配置与代码分离；
- 命令行脚本与核心逻辑分离；
- 数据目录单独说明；
- 输出目录可控；
- 支持逐步增加更多实验配置与模型模块。

## 7. 已实现的功能模块

### 7.1 数据预处理模块

脚本：

```bash
scripts/prepare_data.py
```

当前功能：

- 从指定原始图片目录递归读取图片；
- 支持常见图片格式，如 PNG、JPG、JPEG、WEBP、BMP；
- 将图像统一 resize 到指定尺寸；
- 默认使用 nearest-neighbor，适合 Pixel Art；
- 将图像转换为 RGB；
- 自动生成训练和验证所需的 `jsonl` metadata 文件；
- 支持自动推断简单标签；
- 支持划分 train / val / test。

当前推荐数据组织方式为：

```text
data/raw/pixelart/
data/raw/emoji/
data/processed/
data/metadata/
```

metadata 单行示例：

```json
{"image": "processed/train/000001.png", "text": "red slime", "label": "slime", "source": "pixelart"}
```

这个模块的意义在于：

1. 将后续训练输入格式统一；
2. 将数据组织从“散乱图片文件夹”转换为“可训练元数据格式”；
3. 为后续增加文本编码器、类别控制和多源数据混合打基础。

### 7.2 数据集读取模块

核心文件：

```bash
src/pixelcraft/data/dataset.py
src/pixelcraft/data/transforms.py
```

当前实现了：

- `JsonlImageDataset`
- 图像变换流程；
- 条件 ID 的稳定映射逻辑；
- 图像路径、标签、文本、条件 ID 的统一返回格式。

当前数据读取逻辑采用 `jsonl metadata + image path` 的方式，而不是死绑定某个具体数据集格式。这种设计的优点是：

- 后续可以轻松切换不同来源数据；
- 不需要为每个数据集单独重写训练入口；
- 便于添加 `source`、`style`、`platform` 等额外字段。

### 7.3 条件建模模块

核心文件：

```bash
src/pixelcraft/models/conditioning.py
```

当前实现的是最基础的类别条件：

- 使用 `nn.Embedding` 将类别 ID 映射到条件向量；
- 当前相当于做 label-conditioned generation；
- 后续可以自然扩展到 text encoder。

这是一个刻意保持简化的起点，原因是：

1. 类别条件更容易先跑通；
2. 对数据标注要求更低；
3. 更适合 baseline 验证训练流程是否稳定；
4. 后续增加文本条件时，不需要推倒重来。

### 7.4 扩散模型模块

核心文件：

```bash
src/pixelcraft/models/diffusion.py
```

当前已实现：

- 线性 beta schedule；
- 前向噪声过程 `q_sample`；
- 训练损失 `p_losses`；
- 基础反向采样逻辑；
- 从纯噪声逐步去噪生成图像。

当前设计属于最基本的 DDPM 路线，优点是：

- 逻辑清晰；
- 便于调试；
- 适合课程项目、实验原型和早期研究阶段；
- 可以作为后续加入 DDIM、cosine schedule、v-prediction 等改进的基础。

### 7.5 U-Net 模型模块

核心文件：

```bash
src/pixelcraft/models/unet.py
```

当前实现的是轻量级条件 U-Net，包括：

- Sinusoidal timestep embedding；
- 残差块 `ResBlock`；
- 下采样模块 `DownBlock`；
- 上采样模块 `UpBlock`；
- 与时间 embedding 和条件 embedding 的融合；
- 最终输出噪声预测结果。

这个 U-Net 版本是为低分辨率图像设计的最小可运行结构，重点是：

- 模型足够轻，容易在小图上训练；
- 结构简单，便于理解和修改；
- 适合后续扩展 attention、更多层数、更高通道数。

开发过程中还修复了一个重要实现问题：

- 原先中间层如果直接用 `nn.Sequential`，无法向 `ResBlock` 传递 embedding；
- 之后已修正为显式迭代的模块列表结构，保证时间和条件 embedding 能正确参与前向计算。

### 7.6 训练模块

核心文件：

```bash
scripts/train.py
src/pixelcraft/training/trainer.py
```

当前训练流程已经具备：

- 读取 YAML 配置；
- 固定随机种子；
- 自动选择设备；
- 构建 dataset 和 dataloader；
- 构建 conditioner、U-Net、diffusion；
- 使用 AdamW 优化器；
- 执行 epoch 级训练；
- 记录 loss；
- 定期保存 sample；
- 定期保存 checkpoint。

输出结构为：

```text
outputs/<experiment_name>/
  checkpoints/
  samples/
  logs/
  config.yaml
  source_config.yaml
```

这一设计保证了实验具备基本可复现性：

- 配置会被保存；
- checkpoint 会被保存；
- 训练样例图会被保存；
- loss 日志会被保存。

### 7.7 采样模块

核心文件：

```bash
scripts/sample.py
```

当前功能：

- 从配置文件读取模型结构参数；
- 从 checkpoint 恢复模型和条件嵌入；
- 输入 prompt 字符串；
- 将 prompt 映射到稳定条件 ID；
- 生成指定数量图像；
- 输出图像网格。

虽然当前 prompt 还不是严格意义上的自然语言文本理解，而是通过稳定哈希映射到类别 ID，但这已经足以支撑以下用途：

- 验证采样流程；
- 对固定标签做条件控制；
- 为后续引入文本编码器预留统一接口。

### 7.8 评估模块

核心文件：

```bash
scripts/evaluate.py
```

当前评估模块属于 lightweight sanity evaluation，功能包括：

- 读取指定目录下的生成图片；
- 统计图片数量；
- 统计图片尺寸；
- 计算 RGB 均值和标准差；
- 将结果输出为 JSON。

该模块目前不提供 FID、KID 或 CLIP score。这不是遗漏，而是阶段性取舍：

- 第一阶段重点是验证系统是否能跑通；
- 在正式实验前再接入更重的评估依赖更合理；
- 先用简单统计排查输出是否为空白、是否全黑、是否尺寸异常，会更高效。

此外，评估脚本中还顺手修复了一个过时实现：

- 旧写法涉及 `TypedStorage` 的 warning；
- 当前已改为基于 `numpy` 的更稳定实现。

## 8. 配置文件与实验管理

已提供两个基础配置文件：

```bash
configs/pixelart_32.yaml
configs/emoji_32.yaml
```

配置内容覆盖：

- 实验名称；
- 随机种子；
- 输出目录；
- 数据路径；
- 图像尺寸；
- 模型宽度与层数；
- 条件维度；
- diffusion timestep；
- beta schedule；
- batch size；
- epoch 数；
- 学习率；
- 日志、采样、保存频率。

这说明当前代码已经不是“写死参数”的脚本式原型，而是进入了“配置驱动”的实验框架阶段。

这样做的价值在于：

1. 后续可以快速复制配置做消融实验；
2. 可以轻松区分 Pixel Art 与 Emoji 的实验；
3. 可以更容易保留实验记录；
4. 为多人协作提供统一入口。

## 9. .gitignore 与仓库管理

当前已经添加 `.gitignore`，主要忽略：

- 原始数据；
- 处理后的数据；
- metadata；
- 训练输出；
- checkpoint；
- Python 缓存；
- notebook 缓存。

这样做的目的很明确：

1. 防止误把大文件推到 GitHub；
2. 保持仓库整洁；
3. 让版本控制集中在代码、配置和文档上；
4. 避免训练过程产生的大量临时文件污染主分支。

同时保留了：

```bash
outputs/.gitkeep
```

以便输出目录结构在仓库中有占位存在。

## 10. 端到端验证结果

为了验证框架不是“只写了代码但没运行”，已经执行了一次完整 smoke test。

### 10.1 Smoke Test 内容

执行流程包括：

1. 临时生成少量简单彩色小图作为测试数据；
2. 使用 `prepare_data.py` 生成 `train.jsonl` 和 `val.jsonl`；
3. 构造一个极小配置文件；
4. 运行 1 个 epoch 训练；
5. 导出 checkpoint；
6. 运行采样脚本生成图片；
7. 运行评估脚本生成 JSON 统计结果。

### 10.2 Smoke Test 结果

预处理结果：

```text
Prepared 6 train, 2 val, 0 test images.
```

训练完成：

- 1 个 epoch 正常执行；
- loss 正常下降；
- checkpoint 正常保存；
- sample 图像正常输出。

采样完成：

```text
Saved samples to outputs/smoke/manual_sample.png
Saved samples to outputs/smoke/manual_sample_2.png
```

评估完成，输出示例包括：

- `num_images`
- `unique_sizes`
- `mean_rgb`
- `std_rgb`

这说明当前项目最关键的一点已经成立：

**代码不仅已经搭起来，而且已经可以完整跑通一轮训练和推理流程。**

## 11. 当前已知限制

虽然已经完成基础框架，但当前版本仍然属于 baseline，存在以下明确限制：

### 11.1 条件方式仍然偏简单

当前条件输入本质上还是 label embedding，而不是真正的文本编码器。

影响：

- 暂时不具备真正的 text-to-image 语义理解能力；
- 目前更接近 label-conditioned diffusion。

### 11.2 评估指标较弱

当前评估脚本只做基础 sanity check，没有接入：

- FID
- KID
- CLIP score
- Human preference protocol

### 11.3 数据预处理仍然是通用版

目前还没有针对 Pixel Art 和 Emoji 的特殊预处理细节做深度优化，例如：

- 透明背景策略；
- 调色板分析；
- 类别清洗；
- prompt 标准化；
- 重复图去重。

### 11.4 模型还是最小 baseline

当前模型没有实现：

- attention 模块；
- classifier-free guidance；
- DDIM sampling；
- cosine noise schedule；
- text encoder；
- DiT 架构；
- mixed precision 训练；
- 多卡训练。

### 11.5 尚未完成正式实验

目前完成的是框架和验证，不是正式结果。也就是说：

- 还没有用真实 Pixel Art / Emoji 数据集训练出可展示的最终效果；
- 还没有形成课程报告中的对比实验与定量表格；
- 还没有生成最终汇报需要的高质量可视化结果。

## 12. 下一步建议

基于当前状态，建议后续工作按下面顺序推进。

### 12.1 优先事项一：接入真实数据集

建议先选择一个最容易落地的数据源作为第一版训练集，例如：

- 一个小型 Pixel Art 数据集；
- 或者一个结构简单的 Emoji 数据集。

目标是先用真实数据跑出第一版 baseline 效果图。

### 12.2 优先事项二：完善数据预处理

建议补充：

- 透明背景处理选项；
- 标签清洗；
- 类别映射表；
- prompt 标准化；
- 数据可视化检查脚本。

### 12.3 优先事项三：做第一轮正式训练

建议先做一组小规模实验：

- `32x32`
- 单一数据域
- 小 batch
- 10 到 30 epochs

目标不是立刻做最强结果，而是先拿到一版可分析的 baseline 输出。

### 12.4 优先事项四：升级条件控制

在 label 条件稳定后，可以继续加入：

- 简单文本编码器；
- 更规范的 prompt 到 embedding 过程；
- classifier-free guidance。

### 12.5 优先事项五：补强评估模块

等 baseline 结果稳定后，再引入：

- FID / KID；
- CLIP score；
- 人工打分表；
- prompt 对齐评估。

## 13. 当前结论

截至目前，项目已经顺利完成“从 0 到 1”的基础搭建工作。

更具体地说，已经实现了以下关键目标：

- 服务器环境可用；
- GitHub 协作链路可用；
- 独立 Conda 环境可用；
- GPU 训练环境可用；
- 项目文档清晰；
- 仓库结构清晰；
- 数据预处理可运行；
- 训练流程可运行；
- 采样流程可运行；
- 评估流程可运行；
- 端到端 smoke test 已验证通过。

因此，项目当前已经从“概念阶段”进入“可以开始正式实验的工程阶段”。

后续工作的重点不再是继续堆基础脚手架，而是：

1. 接入真实数据；
2. 跑出第一版 baseline 结果；
3. 逐步增强条件控制与评估；
4. 为最终课程报告或展示材料积累实验结果。
