# Data

本目录用于保存本地数据，不建议把原始数据和训练输出提交到 Git。

推荐结构：

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

每行 metadata 示例：

```json
{"image": "processed/train/000001.png", "text": "red slime", "label": "slime", "source": "pixelart"}
```

可以用下面命令从图片文件夹生成初始数据：

```bash
python scripts/prepare_data.py \
  --input-dir data/raw/pixelart \
  --output-dir data/processed \
  --metadata-dir data/metadata \
  --image-size 32
```
