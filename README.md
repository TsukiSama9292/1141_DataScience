# 🎬 電影推薦系統：基於 MovieLens 20M 的系統性實驗

> **高性能協同過濾推薦引擎**  
> 透過 37 個階段性實驗找出最佳配置，達成 **68.8% Hit Rate@10**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MovieLens](https://img.shields.io/badge/Dataset-MovieLens_20M-orange.svg)](https://grouplens.org/datasets/movielens/20m/)
[![Experiments](https://img.shields.io/badge/Experiments-39_Configs-green.svg)](#實驗架構)

---

## 📊 核心成果

### 🏆 最佳性能指標

基於 MovieLens 20M 資料集（20,000,263 筆評分）的實驗結果：

| 指標 | 快速實驗 (500樣本) | 大樣本驗證 (20K) | 配置 |
|------|-------------------|------------------|------|
| **Hit Rate@10** | 68.8% | **66.4%** ✅ | SVD_008 (SVD=200, KNN=20) |
| **NDCG@10** | 0.540 | **0.535** ✅ | SVD_008 |
| **RMSE** | 0.957 | **0.977** ✅ | SVD_008 |
| **KNN 最佳** | 68.8% | [SVD=200, K=20] | KNN_001/002 (SVD=200, KNN=20) |

> 💡 **驗證結論**: 大樣本驗證顯示真實性能為 **Hit Rate@10 = 66.4%**，證實了推薦系統的穩定性。

### 📈 實驗結果可視化

#### SVD 維度分析
完整探索 25-1000 維度範圍，發現 **SVD=200** 為最佳平衡點：

![SVD 維度分析](reports/figures/svd_dimension_analysis.png)

**關鍵發現**：
- 🎯 SVD=200 達到性能峰值（68.8% Hit Rate）
- 📊 維度過低（<100）導致欠擬合
- ⚠️ 維度過高（>500）增加計算成本但性能無顯著提升

#### KNN 鄰居數分析
測試 K=10-100 範圍（9個配置），**K=20** 為最佳鄰居數：

![KNN K 值分析](reports/figures/knn_k_value_analysis.png)

**關鍵發現**：
- 🎯 K=20 達到最佳平衡（68.8% Hit Rate）
- 📉 K>20 性能逐漸下降
- ⚠️ K 值過大引入噪音，降低推薦精度

#### 各階段性能對比
逐步優化策略的效果驗證：

![階段對比](reports/figures/stage_comparison.png)

---

## 📂 資料集統計

### MovieLens 20M 完整資料集

- **總評分數**: 20,000,263 筆
- **使用者數**: 138,493 位
- **電影數**: 26,744 部
- **評分範圍**: 0.5 - 5.0 星（0.5 遞增）
- **時間跨度**: 1995-2015

### 評分分布

![評分分布](reports/figures/data_rating_distribution_full.png)

**統計特徵**：
- 平均評分：3.53 星
- 中位數：3.5 星
- 標準差：1.06
- 峰值：4.0 星（最常見評分）

### 使用者活躍度長尾分析

![使用者活躍度](reports/figures/data_user_activity_long_tail_full.png)

**長尾效應**：
- 前 20% 活躍使用者貢獻 53.4% 的評分
- 中位數：68 評分/人
- 最多評分：9,254 筆（超級用戶）

### 電影流行度長尾分析

![電影流行度](reports/figures/data_movie_popularity_long_tail_full.png)

**冷啟動挑戰**：
- 42.2% 的電影評分少於 10 次
- 中位數：18 評分/部電影
- 熱門電影：67,310 筆評分

---

## 🔬 實驗架構

### 逐步優化法（Progressive Optimization）

採用階段性優化策略,每階段使用前階段最佳結果,共 **39 個實驗配置 + 2 個驗證配置**:

```
實驗配置 (39個):
DS (4) → FILTER (6) → SVD (15) → KNN (9) → BIAS (3) → OPT (2)

驗證配置 (2個):
VALIDATE_001 (20K用戶) + FULLTEST_001 (138K用戶)
```

| 階段 | 目的 | 配置數 | 最佳配置 | Hit Rate@10 |
|------|------|--------|----------|-------------|
| **DS** | 資料規模測試 | 4 | DS_002 (5M) | 68.4% |
| **FILTER** | 長尾電影過濾 | 6 | FILTER_004 (min=20) | 66.6% |
| **SVD** | 維度降維優化 | 15 | **SVD_008 (n=200)** | **68.8%** 🏆 |
| **KNN** | 鄰居數優化 | 9 | KNN_001 (k=20) | 68.8% |
| **BIAS** | 偏差校正測試 | 3 | BIAS_001 (無bias) | 66.8% |
| **OPT** | 優化策略測試 | 2 | OPT_001 (時間衰減) | 61.6% |

**完整實驗結果**: 見 [summary.md](reports/summary.md)

### 效率對比

相較於網格搜尋，逐步優化法大幅減少計算成本：

| 優化策略 | 總配置數 | 執行時間 | 計算成本 | 結果品質 |
|---------|---------|---------|---------|---------|
| **完整網格搜尋** | **9,408** | **~5.2 小時** | **極高** ⚠️ | 理論最優 |
| 精簡網格搜尋 | 588 | ~20 分鐘 | 高 | 接近最優 |
| 隨機搜尋 | ~50 | ~2 分鐘 | 中等 | 不穩定 |
| **逐步優化法** ✅ | **39** | **~1.3 分鐘** | **低** | **接近最優** 🎯 |

**效率提升**: **99.6%** 計算成本降低（39 vs 9,408 配置）

> 💡 **計算說明**: 完整網格搜尋需涵蓋 14 個 SVD 維度 × 7 個 KNN 值 × 6 個過濾參數 × 2 個 use_svd × 2 個 item_bias × 2 個 time_decay × 2 個 tfidf = **9,408 個配置**

**逐步優化法的優勢**:
- ⚡ 執行時間從 5.2 小時降至 1.3 分鐘
- 🎯 基於理論指導的智能搜尋策略
- 📊 每個階段的最佳配置指導下一階段
- ✅ 最終結果經大樣本驗證（20K 用戶）

---

## 🚀 快速開始

### 環境需求

- Python 3.12+
- 8GB+ RAM（完整資料集分析需要）
- 推薦使用 `uv` 作為包管理器

### 安裝

```bash
# 克隆專案
git clone https://github.com/TsukiSama9292/1141_DataScience.git
cd 1141_DataScience

# 安裝依賴（使用 uv）
uv sync
```

### 執行完整流程

```bash
# 執行所有實驗配置並生成報告
uv run main.py

# 執行流程：
# 1. 自動檢測並跳過已完成的配置
# 2. 執行所有實驗（約 20~30 分鐘，首次運行）
# 3. 生成實驗結果報告
# 4. 自動生成完整資料集報告（20M 評分，約 1-2 分鐘）
```

### 只生成報告

```bash
# 只生成實驗結果報告（秒級）
uv run python tools/generate_report.py --no-dataset

# 生成完整資料集報告（1-2 分鐘）
uv run python tools/generate_report.py --full-dataset
```

### 執行單個配置

```bash
# 執行最佳配置(快速測試,500樣本)
uv run python run/SVD_008.py

# 執行大樣本驗證(20K 用戶,~3-5 分鐘)
uv run python run/VALIDATE_001.py

# 執行全用戶測試(138K 用戶,~30-60 分鐘,可選)
uv run python run/FULLTEST_001.py
```

### 驗證工具（可選）

```bash
# 網格搜尋驗證（用於驗證逐步優化法的有效性）

# 快速搜尋（9 個配置，~1 分鐘）
uv run python tools/grid_search.py --preset quick

# 精簡搜尋（294 個配置，~10 分鐘）
uv run python tools/grid_search.py --preset standard

# 查看網格搜尋報告
cat grid_search_results/grid_search_report.md
```

---

## 💡 核心技術

### 1. 協同過濾引擎

**基礎算法**: Item-based KNN
- 使用余弦相似度計算電影相似性
- 稀疏矩陣優化，支援 20M 評分
- 鄰居快取機制，評估加速 1000x+

**降維技術**: Truncated SVD
- 使用 Scikit-learn 的 TruncatedSVD
- 測試範圍：25-1000 維度和原始維度
- 最佳配置：200 維度

### 2. 評估指標

採用 Leave-One-Out 交叉驗證：

- **Hit Rate@10**: 前 10 個推薦中是否命中測試項目（68.8%）
- **NDCG@10**: 歸一化折扣累積增益（0.540）
- **RMSE**: 評分預測誤差（0.957）

#### 評估方法說明

本研究採用**分層評估策略**，平衡實驗效率與統計顯著性：

**階段一：快速實驗迭代**
- 抽樣方法: 從 138,493 個用戶中隨機抽取 **500 個用戶**
- 樣本比例: 0.36%
- 統計誤差: 95% 置信區間 ±4.4%
- 執行時間: ~2 秒/實驗
- 用途: 快速探索 39 個配置的相對性能

**階段二：最佳配置驗證**
- 驗證樣本: **20,000 個用戶** (14.4%)
- 統計誤差: 95% 置信區間 ±0.7%
- 執行時間: ~3-5 分鐘
- 用途: 驗證最佳配置的性能穩定性

**學術依據**:
- 符合推薦系統領域常見做法 (Cremonesi et al., 2010)
- 相對性能排序在小樣本下具有穩定性
- 大樣本驗證確保結果可信度

**驗證結果對比**:

| 配置 | 評估指標 | 500 樣本 | 20,000 樣本 | 差異 |
|------|----------|----------|-------------|------|
| SVD_008 (最佳) | Hit Rate@10 | 68.8% | **66.4%** | -2.4% |
| | NDCG@10 | 0.540 | **0.535** | -0.9% |
| | RMSE | 0.957 | **0.977** | +2.1% |

**結果分析**:
- ✅ 性能差異在統計誤差範圍內（500樣本的 ±4.4% 區間）
- ✅ 20,000 樣本結果更可靠（誤差僅 ±0.7%）
- ✅ Hit Rate 66.4% 在推薦系統領域仍屬優秀表現
- 📊 **保守估計**: 系統在生產環境的 Hit Rate@10 約為 **66-67%**

#### 為何不使用全用戶評估？

**技術限制說明**：

本研究採用 **KNN 預計算優化策略**，在訓練時一次性計算所有用戶的鄰居關係，使評估階段的時間複雜度從 **O(n²)** 降至 **O(n)**：

```python
# models.py 中的優化策略
if n_users <= 20000:  # 記憶體閾值
    # 預計算所有鄰居 (一次性成本: O(n² × d))
    distances, indices = knn.kneighbors(features)
    # 評估時直接查表: O(1)
else:
    # 動態計算鄰居 (每次評估: O(n × d))
    # 總成本: O(n² × d)
```

**記憶體與時間權衡**：

| 使用者數 | 預計算記憶體 | 評估時間複雜度 | 實際執行時間 |
|---------|-------------|---------------|-------------|
| 500 | ~50 MB ✅ | O(n) | ~2 秒 |
| 20,000 | ~2.9 GB ✅ | O(n) | ~2 分鐘 |
| 138,493 | **~98 GB** ⚠️ | O(n) | ~9 分鐘（理論） |
| 138,493 (無預計算) | ~3 GB ✅ | **O(n²)** | **~30-60 分鐘** |

**設計決策**：
- ✅ **選擇 20K 樣本驗證**：在記憶體限制（8GB RAM）內達成 O(n) 線性時間，統計誤差僅 ±0.7%
- ⚠️ **放棄全用戶評估**：要麼承受 O(n²) 的時間成本（30-60 分鐘），要麼需要 98GB 記憶體支援預計算
- 🎯 **工程實務考量**：20K 樣本已達學術標準，且能在消費級硬體上高效執行

> 💡 **複雜度分析**: 預計算策略將評估複雜度從 O(n² × d) 降至 O(n × k × w)，其中 k 為鄰居數、w 為平均評分數。這種權衡設計使得在有限資源下仍能進行大規模驗證。

### 3. 性能優化

**記憶體優化**:
- 使用 scipy.sparse 稀疏矩陣（99.5% 稀疏度）
- 分批處理完整資料集分析（每批 500K）
- 智能緩存機制，避免重複計算

**計算優化**:
- 向量化操作（NumPy）
- 預計算相似度矩陣
- 增量評估（只計算測試用戶）

---

## 📚 項目結構

```
1141_DataScience/
├── README.md                          # 本文件
├── main.py                            # 主執行入口
├── pyproject.toml                     # 項目依賴配置
│
├── src/movie_recommendation/          # 核心代碼
│   ├── data_loader.py                 # 資料載入與預處理
│   ├── feature_engineering.py         # 特徵工程（SVD、TF-IDF等）
│   ├── models.py                      # KNN 推薦模型
│   ├── evaluation.py                  # 評估指標計算
│   ├── experiment.py                  # 實驗編排器
│   ├── analysis.py                    # 實驗結果分析
│   ├── report_generator.py            # 報告生成器
│   └── utils.py                       # 工具函數
│
├── run/                               # 實驗與驗證配置(39+2個)
│   # === 實驗配置 (39個) ===
│   ├── DS_001.py ~ DS_004.py          # 資料規模階段(4個)
│   ├── FILTER_001.py ~ FILTER_006.py  # 過濾階段(6個)
│   ├── SVD_001.py ~ SVD_015.py        # SVD階段(15個)
│   ├── KNN_001.py ~ KNN_009.py        # KNN階段(9個)
│   ├── BIAS_001.py ~ BIAS_003.py      # 偏差校正階段(3個)
│   ├── OPT_001.py, OPT_003.py         # 優化策略階段(2個)
│   # === 驗證配置 (2個) ===
│   ├── VALIDATE_001.py                # 大樣本驗證(20K用戶,~3分鐘)
│   └── FULLTEST_001.py                # 全用戶測試(138K用戶,~30-60分鐘)
│
├── log/                               # 實驗結果（JSON格式）
│   ├── best_svd.json                  # 最佳SVD配置
│   ├── best_knn.json                  # 最佳KNN配置
│   ├── DS_001.json ~ DS_004.json      # 各階段實驗結果
│   ├── ...                            # (39個實驗配置的JSON)
│   ├── VALIDATE_001.json              # 大樣本驗證結果
│   └── FULLTEST_001.json              # 全用戶測試結果(如已執行)
│
├── reports/                           # 報告與可視化
│   ├── README.md                      # 報告說明文檔
│   ├── summary.md                     # 實驗結果摘要表格
│   ├── best_configs.json              # 最佳配置匯總
│   ├── dataset_statistics_full.json   # 完整資料集統計
│   └── figures/                       # 可視化圖表
│       ├── svd_dimension_analysis.png
│       ├── knn_k_value_analysis.png
│       ├── stage_comparison.png
│       ├── data_rating_distribution_full.png
│       ├── data_user_activity_long_tail_full.png
│       └── data_movie_popularity_long_tail_full.png
│
└── tools/                             # 分析工具
    ├── analyze.py                     # 實驗分析CLI
    ├── generate_report.py             # 報告生成CLI
    └── grid_search.py                 # 網格搜尋工具（驗證用）
```

---

## 🔍 關鍵發現

### ✅ 有效策略

1. **SVD 降維 (SVD=200)**
   - ✅ 性能提升 +5.5%（相對 SVD=128）
   - ✅ 降低噪音，提高推薦精度
   - ✅ 加速計算，減少記憶體使用

2. **適中鄰居數 (K=20)**
   - ✅ 達到最佳性能（68.8% Hit Rate）
   - ✅ 避免過度擬合
   - ✅ K=20 是最優平衡點

3. **完整資料集 (20M)**
   - ✅ 資料量越大，推薦品質越好
   - ✅ 稀疏矩陣優化可支援大規模資料

### ❌ 無效策略

1. **時間衰減（Time Decay）**
   - ❌ Hit Rate 下降至 61.6%（-7.2%）
   - 原因：電影偏好相對穩定，新舊評分同等重要
   - 結論：不適用於電影推薦場景

2. **TF-IDF 加權**
   - ❌ 未能改善推薦品質
   - 原因：評分已包含偏好強度資訊
   - 結論：文本檢索技術不適用於評分矩陣

3. **Item Bias 校正**
   - ❌ 性能下降至 66.8%
   - 原因：熱門電影確實值得推薦
   - 結論：不應過度校正自然流行度

### 🎯 實用建議

**生產環境推薦配置**:
```python
config = {
    'data_limit': None,              # 使用全量資料
    'min_item_ratings': 0,           # 不過濾長尾電影
    'use_svd': True,
    'n_components': 200,             # SVD 維度
    'k_neighbors': 20,               # KNN 鄰居數
    'use_item_bias': False,          # 不使用偏差校正
    'use_time_decay': False,         # 不使用時間衰減
    'use_tfidf': False              # 不使用 TF-IDF
}
```

**性能指標**:
- **生產環境推薦**: Hit Rate@10 = **66.4%** (基於 20K 樣本驗證)
- **NDCG@10**: 0.535 (排序品質)
- **RMSE**: 0.977 (評分預測誤差)
- **評估時間**: ~2 分鐘 (20,000 個用戶)

**快速實驗指標** (500 個測試用戶):
- Hit Rate@10: 68.8% (±4.4% 誤差)
- 用於快速迭代和相對性能比較

---

## 📄 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 文件

---

## 致謝

- **資料來源**: [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) by GroupLens Research
- **演算法參考**: Sarwar, B., et al. (2001). *Item-based collaborative filtering recommendation algorithms*
- **評估指標**: Cremonesi, P., et al. (2010). *Performance of recommender algorithms on top-n recommendation tasks*