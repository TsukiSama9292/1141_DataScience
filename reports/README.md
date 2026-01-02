# 實驗報告說明

本目錄包含電影推薦系統的完整實驗報告和數據分析。

## 文件結構

### 實驗結果報告

- `summary.md` - 實驗結果摘要表格（各階段最佳配置）
- `best_configs.json` - 最佳 SVD 和 KNN 配置詳情
- `figures/svd_dimension_analysis.png` - SVD 維度分析圖
- `figures/knn_k_value_analysis.png` - KNN K 值分析圖
- `figures/stage_comparison.png` - 各階段性能對比圖

### 資料集分析報告

資料集分析文件使用後綴來標識數據量：

- `_full` - 完整資料集（20,000,263 筆評分）
- `_100000` - 樣本資料（100,000 筆評分）
- 無後綴 - 舊格式（已棄用）

#### 完整資料集報告（推薦使用）

- `dataset_statistics_full.json` - 完整資料集統計數據
  - 138,493 位使用者
  - 26,744 部電影
  - 20,000,263 筆評分

- `figures/data_rating_distribution_full.png` - 評分分布圖（0.5-5.0 星）
- `figures/data_user_activity_long_tail_full.png` - 使用者活躍度長尾分析
- `figures/data_movie_popularity_long_tail_full.png` - 電影流行度長尾分析

## 報告生成

### 自動生成（推薦）

運行主程序會自動生成所有報告：

```bash
uv run main.py
```

執行流程：
1. 執行所有實驗配置（自動跳過已完成）
2. 生成實驗結果報告
3. 檢查並生成完整資料集報告（如果不存在）

### 手動生成

#### 只生成實驗結果報告

```bash
uv run python tools/generate_report.py --no-dataset
```

#### 生成樣本資料集報告（快速，約 10 秒）

```bash
uv run python tools/generate_report.py
```

#### 生成完整資料集報告（完整，約 1-2 分鐘）

```bash
uv run python tools/generate_report.py --full-dataset
```

## 智能緩存機制

報告系統會自動檢測已存在的文件：

- ✅ 如果資料集圖表和統計文件已存在，自動跳過生成
- 📊 節省時間：完整資料集分析只需執行一次
- 🔄 如需重新生成，手動刪除對應文件即可

### 強制重新生成完整資料集報告

```bash
# 刪除完整資料集文件
rm reports/dataset_statistics_full.json
rm reports/figures/data_*_full.png

# 重新生成
uv run python tools/generate_report.py --full-dataset
```

## 性能優化

完整資料集分析使用分批處理（chunking）技術：

- 每批處理 500,000 筆評分
- 自動顯示處理進度
- 內存占用優化，避免 OOM
- 總處理時間約 1-2 分鐘

## 數據來源

- **資料集**: MovieLens 20M Dataset
- **來源**: GroupLens Research (via Kaggle)
- **大小**: 20M 評分、138K 使用者、27K 電影
- **時間範圍**: 1995-2015

## 查看報告

所有圖表為 PNG 格式，可直接在瀏覽器或圖片查看器中打開。
統計數據為 JSON 格式，建議使用 `jq` 或文本編輯器查看。

```bash
# 查看統計摘要
cat reports/summary.md

# 查看完整資料集統計（格式化）
cat reports/dataset_statistics_full.json | jq .

# 查看最佳配置
cat reports/best_configs.json | jq .
```
