# Pixelcraft-Diffusion

低分辨率视觉内容生成项目，目标是基于扩散模型生成 **32x32 / 64x64** 的像素画、Emoji 和小图标。项目会尽量保留扩散模型的核心技术栈，包括噪声调度、U-Net / Diffusion Transformer、条件生成、采样策略和评估流程，同时控制训练成本，使其可以在单张入门级 GPU 或 Google Colab 环境中运行。

## 项目目标

本项目关注“低分辨率 + 风格化 + 条件控制”的图像生成任务。

给定类别标签或文本描述，模型需要生成对应的小尺寸视觉内容，例如：

| 方向 | 生成目标 | 分辨率 | 条件输入示例 |
| --- | --- | --- | --- |
| Pixel Art | 游戏风格角色、道具、场景 | 32x32 / 64x64 | `Red Slime`, `Blue Sword` |
| Emoji / Icon | 表情、小图标、符号 | 32x32 / 64x64 | `Smiley face`, `Flame`, `Heart` |

阶段性目标：

1. 搭建可运行的数据处理、训练、采样和评估框架。
2. 先完成类别条件生成或短文本条件生成。
3. 在小数据集上训练 baseline diffusion model。
4. 对比不同分辨率、噪声调度、网络结构和采样步数的效果。
5. 输出可复现实验结果，包括模型权重、生成样例和实验记录。

## 推荐技术路线

### 1. 数据准备

优先选择规模适中、标注简单、图像风格明确的数据集。

Pixel Art 可选数据集：

| 数据集 | 说明 | 来源 |
| --- | --- | --- |
| DiffusionDB-PixelArt | 约 2,000 张图像，带文本 prompt | HuggingFace |
| free-to-use-pixelart | 多张像素图，可能带尺寸或类别信息 | HuggingFace |
| Pixel Art Dataset | 多类别像素画图像 | Kaggle |

Emoji / Icon 可选数据集：

| 数据集 | 说明 | 来源 |
| --- | --- | --- |
| Full Emoji Image Dataset | 多平台 Emoji 图像 | Kaggle |
| valhalla/emoji-dataset | 约 2,749 张图像，带文本描述 | HuggingFace |
| Noto Emoji Dataset | Google Noto Emoji 图像 | HuggingFace |

建议统一处理为如下格式：

```text
data/
  raw/
    pixelart/
    emoji/
  processed/
    train/
    val/
    test/
  metadata/
    train.jsonl
    val.jsonl
    test.jsonl
```

`metadata/*.jsonl` 示例：

```json
{"image": "processed/train/000001.png", "text": "red slime", "label": "slime", "source": "pixelart"}
{"image": "processed/train/000002.png", "text": "smiley face", "label": "smile", "source": "emoji"}
```

数据预处理建议：

1. 统一分辨率为 `32x32` 或 `64x64`。
2. 保持像素画边缘清晰，resize 时优先使用 nearest-neighbor。
3. 统一背景处理，透明背景可转为白色、黑色或 alpha mask。
4. 清理重复图、严重水印图和分辨率过低的样本。
5. 划分训练集、验证集和测试集，并固定随机种子。

### 2. 模型设计

Baseline 优先实现轻量级 DDPM：

```text
condition text / label
        |
condition encoder
        |
noise image x_t + timestep embedding
        |
small U-Net / small DiT
        |
predicted noise epsilon
```

推荐从简单版本开始：

1. 图像尺寸：先使用 `32x32`，稳定后扩展到 `64x64`。
2. 模型结构：先使用小型 U-Net。
3. 条件方式：先做 label embedding，再做 text embedding。
4. 训练目标：预测噪声 `epsilon`。
5. 噪声调度：先使用 linear beta schedule，再尝试 cosine schedule。
6. 采样方式：先实现 DDPM sampling，再扩展 DDIM sampling。

### 3. 训练流程

建议训练命令形式：

```bash
python scripts/train.py \
  --config configs/pixelart_32.yaml
```

训练过程应记录：

1. loss 曲线。
2. 固定 prompt 的周期性采样结果。
3. checkpoint。
4. config 文件快照。
5. 当前 git commit hash。

推荐输出结构：

```text
outputs/
  pixelart_32/
    checkpoints/
    samples/
    logs/
    config.yaml
```

### 4. 推理与展示

建议推理命令形式：

```bash
python scripts/sample.py \
  --checkpoint outputs/pixelart_32/checkpoints/latest.pt \
  --prompt "red slime" \
  --num-images 16 \
  --output outputs/demo/red_slime.png
```

展示内容建议：

1. 不同 prompt 的生成网格图。
2. 训练过程中同一 prompt 的演化结果。
3. 32x32 与 64x64 的对比。
4. 不同 sampling steps 的速度与质量对比。

### 5. 评估方式

低分辨率风格化图像不一定适合只看 FID，建议结合定量和人工评估。

可选指标：

1. FID / KID：衡量整体分布差异。
2. CLIP score：衡量文本和图像匹配程度。
3. Diversity：同一 prompt 下生成结果的多样性。
4. Human preference：人工比较清晰度、风格一致性和语义匹配。

## 建议仓库结构

后续代码可以按下面结构逐步搭建：

```text
Pixelcraft-Diffusion/
  README.md
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
        dataset.py
        transforms.py
      models/
        unet.py
        diffusion.py
        conditioning.py
      training/
        trainer.py
        losses.py
      utils/
        config.py
        image.py
        seed.py
  notebooks/
    exploration.ipynb
  outputs/
    .gitkeep
```

说明：

1. `configs/` 保存实验配置，方便复现实验。
2. `scripts/` 保存命令行入口。
3. `src/pixelcraft/` 保存核心代码。
4. `data/` 和 `outputs/` 默认不提交大文件，只保留说明或 `.gitkeep`。
5. 模型权重、数据集和大量生成图应使用网盘、HuggingFace 或 GitHub Release 管理。

## GitHub 协作规范

### 分支策略

主分支：

1. `main`：稳定版本，只合并经过检查的代码。
2. 功能分支：每个任务单独开分支。

分支命名建议：

```text
feature/data-pipeline
feature/unet-baseline
feature/train-script
feature/sampling
feature/evaluation
fix/readme-typo
experiment/cosine-schedule
```

### 提交流程

推荐流程：

```bash
git checkout main
git pull origin main
git checkout -b feature/your-task

# 修改代码

git status
git add .
git commit -m "Add dataset preprocessing pipeline"
git push origin feature/your-task
```

然后在 GitHub 上创建 Pull Request。

### Commit Message 规范

建议使用简短清晰的英文 commit message：

```text
Add README project framework
Implement pixel art dataset loader
Add DDPM noise scheduler
Fix sample image grid saving
Update emoji preprocessing config
```

常见前缀也可以使用：

```text
docs: update project workflow
feat: add baseline unet
fix: handle transparent emoji images
exp: add cosine noise schedule config
```

### Pull Request 要求

每个 PR 建议包含：

1. 修改内容概述。
2. 如何运行或验证。
3. 关键实验结果或截图。
4. 是否引入新依赖。
5. 是否影响已有训练结果。

PR 描述模板：

```markdown
## Summary

- 

## How to Test

- 

## Results

- 

## Notes

- 
```

### 代码评审重点

Review 时优先检查：

1. 是否能复现运行。
2. 配置是否写死路径。
3. 数据和输出文件是否错误提交。
4. 随机种子是否固定。
5. 训练日志和 checkpoint 路径是否清晰。
6. 代码是否和现有结构一致。

### 文件提交规则

建议提交：

1. 源代码。
2. 小型配置文件。
3. 文档。
4. 小尺寸示例图。
5. 可复现实验脚本。

不建议提交：

1. 原始数据集。
2. 大模型权重。
3. 大量训练输出图。
4. 本地虚拟环境。
5. 缓存文件，如 `__pycache__`、`.ipynb_checkpoints`。

后续应在 `.gitignore` 中忽略：

```gitignore
data/raw/
data/processed/
outputs/
checkpoints/
wandb/
__pycache__/
.ipynb_checkpoints/
*.pt
*.pth
*.ckpt
```

## 推荐开发里程碑

### Milestone 1: 项目初始化

目标：完成 README、仓库结构、环境依赖和数据说明。

产出：

1. 中文 README。
2. 基础目录结构。
3. `requirements.txt`。
4. 数据下载和预处理说明。

### Milestone 2: 数据管线

目标：完成数据集读取、resize、metadata 生成和 dataloader。

产出：

1. `prepare_data.py`。
2. `PixelArtDataset` / `EmojiDataset`。
3. 可视化数据样例网格。

### Milestone 3: Baseline DDPM

目标：完成可训练的小型 U-Net DDPM。

产出：

1. U-Net 模型。
2. diffusion scheduler。
3. training loop。
4. 训练 loss 曲线。

### Milestone 4: 条件生成

目标：支持 label 或 text 条件输入。

产出：

1. label embedding。
2. text embedding 或 prompt encoder。
3. conditional sampling demo。

### Milestone 5: 评估与展示

目标：整理结果，形成报告和演示材料。

产出：

1. 指标结果。
2. 生成样例图。
3. 消融实验。
4. 项目报告。

## 参考资料

1. Latent Diffusion Models: https://arxiv.org/abs/2112.10752
2. Stable Diffusion: https://github.com/compvis/stable-diffusion
3. HuggingFace Diffusers text-to-image example: https://github.com/huggingface/diffusers/tree/main/examples/text_to_image

## 当前状态

当前仓库处于项目初始化阶段。下一步建议先补充 `.gitignore`、`requirements.txt`、基础目录结构和第一个数据预处理脚本。
