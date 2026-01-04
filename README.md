# 🎬 電影推薦系統：MovieLens 20M 協同過濾實驗

> **大規模推薦系統實驗平台**  
> 透過 235+ 組系統性實驗，探索 SVD 降維與 KNN 協同過濾的最佳配置

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MovieLens](https://img.shields.io/badge/Dataset-MovieLens_20M-orange.svg)](https://grouplens.org/datasets/movielens/20m/)
[![Experiments](https://img.shields.io/badge/Experiments-235+_Configs-green.svg)](#實驗架構)

---

## 📊 核心成果

### 🏆 最佳性能指標

基於 MovieLens 20M 資料集（20,000,263 筆評分，138,493 位使用者）的實驗結果：

> 📝 **術語說明**: Hit Rate@10 中的「@10」表示推薦列表長度（top-N 中的 N=10），與下文的 KNN 鄰居數量（k_neighbors）是不同的參數。

| 模型類型 | 最佳配置 | Hit Rate@10 | NDCG@10 | RMSE | 相對提升 |
|---------|---------|------------|---------|------|---------|
| **SVD + KNN** ✅ | SVD=1024, K=40 鄰居 | **67.74%** | **0.5363** | **0.9600** | 基準 |
| **純 KNN** | K=35 鄰居 | 67.67% | 0.5324 | 0.9507 | -0.1% |

> 💡 **關鍵發現**: SVD 降維對性能影響甚微（+0.1%），但在高維度（1024）時略優於純 KNN。純 KNN 在中等鄰居數量（K=35）時已達接近最優性能。

### 📈 實驗結果可視化

#### 1. SVD + KNN 網格搜索熱力圖

完整探索 10 種 SVD 維度 × 10 種 KNN 值 = 100 個配置：

![SVD×KNN 網格搜索](reports/figures/svd_knn_grid_heatmap.png)

**關鍵發現**：
- 🎯 **最佳配置**: SVD=1024, K=40 鄰居（Hit Rate@10 = 67.74%）
- 📊 **高維度優勢**: SVD≥512 時性能穩定，1024 維達到峰值
- 🔄 **鄰居數量敏感性**: K=35-45 鄰居為最佳範圍，過高或過低都會降低性能
- ⚡ **性能平台**: 多個配置達到接近最優性能（67.4%-67.7%）

**前10名配置**：
| 排名 | 配置（SVD維度, KNN鄰居數） | Hit Rate@10 | 說明 |
|-----|------|----------|------|
| 1 | SVD=1024, K=40 | 67.74% | 🏆 全局最優 |
| 2 | SVD=1024, K=45 | 67.73% | 接近最優 |
| 3 | SVD=1024, K=35 | 67.67% | 接近最優 |
| 4 | SVD=1024, K=50 | 67.60% | 仍然優秀 |
| 5 | SVD=512, K=35 | 67.54% | 中等維度最佳 |

#### 2. SVD 維度分析

測試範圍：2 ~ 1024 維度（包含原始高維）

![SVD 維度分析](reports/figures/svd_dimension_analysis.png)

**關鍵洞察**：
- 📉 **低維度瓶頸**: SVD<100 時性能顯著下降（欠擬合）
- 📈 **維度收益遞減**: SVD=512 後提升幅度<0.5%
- 🎯 **實用平衡點**: SVD=512（性能 67.54% vs 計算成本 ↓50%）
- ⚡ **極限性能**: SVD=1024 達到峰值但計算成本倍增

#### 3. KNN 鄰居數量分析（純 KNN 基準線）

測試範圍：K=5 ~ 50 鄰居（純 KNN，無 SVD 降維）

![KNN K 值分析](reports/figures/knn_k_value_analysis.png)

**關鍵洞察**：
- 🎯 **最佳鄰居數量**: K=35（Hit Rate@10 = 67.67%）
- 📊 **性能曲線**: K=5 → 35 快速提升，K>35 緩慢下降
- ⚠️ **過多鄰居**: K>40 引入噪音，性能下降
- 💡 **實用範圍**: K=25-40 鄰居為穩健選擇（Hit Rate@10: 67.25%-67.67%）

**純 KNN 詳細結果**（推薦列表長度 N=10）：
| K 鄰居數 | Hit Rate@10 | NDCG@10 | 相對最優 |
|------|----------|------|---------|
| 5 | 62.18% | 0.4817 | -8.8% |
| 10 | 65.48% | 0.5089 | -3.3% |
| 15 | 66.94% | 0.5220 | -1.1% |
| 20 | 67.25% | 0.5270 | -0.6% |
| 25 | 67.44% | 0.5299 | -0.3% |
| 30 | 67.53% | 0.5312 | -0.2% |
| **35** | **67.67%** | **0.5324** | **基準** 🏆 |
| 40 | 67.60% | 0.5319 | -0.1% |
| 45 | 67.55% | 0.5314 | -0.2% |
| 50 | 67.48% | 0.5304 | -0.3% |

#### 4. SVD + KNN 擴展網格搜索

探索極限性能：13 種 SVD 維度（2-8192）× 9 種 KNN 值（40-80）= 117 個配置

![SVD×KNN 擴展搜索](reports/figures/svd_knn_expand_heatmap.png)

**關鍵發現**：
- 🚀 **高維度極限**: SVD=1024-8192 在 K=40 鄰居時達到最優
- 📊 **低維度表現**: SVD<512 時性能明顯下降
- 🔄 **鄰居數飽和**: K>50 後性能無提升甚至下降
- ✅ **驗證結果**: 確認 SVD=1024, K=40 鄰居為最佳配置

#### 5. 各階段最佳配置對比

![階段對比](reports/figures/stage_comparison.png)

**階段演進分析**：
| 階段 | 最佳配置 | Hit Rate | 提升幅度 | 關鍵改進 |
|------|---------|----------|---------|---------|
| FILTER | min_ratings=20 | 67.27% | 基準 | 過濾長尾電影 |
| KNN_BASELINE | K=35 | 67.67% | +0.6% | 純 KNN 優化 |
| SVD_KNN_GRID | SVD=1024, K=40 | **67.74%** | **+0.7%** | SVD+KNN 聯合優化 |
| BIAS | 無偏差校正 | 67.74% | +0.0% | 偏差校正無效 |
| OPT | 時間衰減 | 66.14% | -2.4% | ❌ 性能下降 |

---

## 📂 資料集統計

### MovieLens 20M 資料集概況

> 📊 **資料來源**: 以下統計基於 **完整 20M 資料集**，使用分批處理以優化記憶體使用。

- **總評分數**: 20,000,263 筆
- **使用者數**: 138,493 位
- **電影數**: 26,744 部
- **評分範圍**: 0.5 - 5.0 星（0.5 遞增）
- **時間跨度**: 1995-2015
- **稀疏度**: 99.5%（極度稀疏）

### 評分分布特徵

![評分分布](reports/figures/data_rating_distribution_full.png)

**統計特性**：
- **平均評分**: 3.53 星
- **中位數**: 3.5 星
- **標準差**: 1.05
- **峰值**: 4.0 星（最常見評分）
- **分布型態**: 左偏（高評分較多）

### 使用者活躍度長尾分析

![使用者活躍度](reports/figures/data_user_activity_long_tail_full.png)

**長尾效應**：
- **頭部使用者**（前 20%）貢獻 63.2% 的評分
- **中位數**: 68 評分/人
- **平均值**: 144 評分/人
- **最多評分**: 9,254 筆（超級使用者）
- **活躍使用者門檻**: 334 評分（前 10%）

### 電影流行度長尾分析

![電影流行度](reports/figures/data_movie_popularity_long_tail_full.png)

**冷啟動挑戰**：
- **冷門電影**（≤5 評分）佔 34.5%
- **中位數**: 18 評分/部電影
- **平均值**: 748 評分/部電影
- **熱門電影**: 最高 67,310 筆評分
- **推薦難度**: 重要知識 - 1/3 電影缺乏足夠訊號

---

## 🔬 實驗架構

### 系統性實驗設計

採用**多階段網格搜索**策略，共 **235+ 個實驗配置**：

```
階段式探索：
FILTER (6) → KNN_BASELINE (10) → SVD_KNN_GRID (100) → SVD_KNN_EXPAND (117) → BIAS (2) → OPT (2)
```

| 階段 | 目的 | 配置數 | 最佳配置 | Hit Rate@10 |
|------|------|--------|----------|-------------|
| **FILTER** | 長尾電影過濾測試 | 6 | min_ratings=20 | 67.27% |
| **KNN_BASELINE** | 純 KNN 基準線 | 10 | K=35 | **67.67%** |
| **SVD_KNN_GRID** | SVD×KNN 網格搜索 | 100 | SVD=1024, K=40 | **67.74%** 🏆 |
| **SVD_KNN_EXPAND** | 擴展維度搜索 | 117 | SVD=1024, K=40 | **67.74%** |
| **BIAS** | 偏差校正測試 | 2 | 無偏差校正 | 67.74% |
| **OPT** | 時間衰減測試 | 2 | 時間衰減 | 66.14% ❌ |

**完整實驗結果**: 見 [summary.md](reports/summary.md)

### 實驗方法學

**網格搜索範圍**：
- **SVD 維度**: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] + 擴展 [2048, 4096, 8192]
- **KNN 鄰居數量**: K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] + 擴展 [40-80, 每5]
- **推薦列表長度**: 固定 N = 10（用於 Hit Rate@10 和 NDCG@10 計算）
- **過濾閾值**: [0, 5, 10, 20, 50, 100]
- **優化策略**: [時間衰減, TF-IDF, 偏差校正]

**評估協議**：
- **訓練/測試分割**: Leave-One-Out（每使用者保留1個測試項）
- **評估樣本**: 20,000 個隨機使用者（14.4% 全使用者）
- **推薦列表長度**: N = 10（固定）
- **評估指標**: Hit Rate@10, NDCG@10, RMSE
- **重複性**: 固定隨機種子（random_state=42）

---

## 💡 核心技術

### 1. 協同過濾引擎

**基礎算法**: Item-based KNN（基於項目的協同過濾）

```python
# 核心相似度計算
from sklearn.neighbors import NearestNeighbors

# 使用餘弦相似度
model = NearestNeighbors(
    n_neighbors=k_neighbors,
    metric='cosine',
    algorithm='brute'  # 稀疏矩陣優化
)
model.fit(item_features)  # 電影特徵矩陣（使用者×評分）
```

**技術亮點**：
- ✅ 稀疏矩陣優化（scipy.sparse.csr_matrix）
- ✅ 向量化計算（避免 Python 迴圈）
- ✅ 預計算相似度矩陣（評估加速 100x+）

**降維技術**: Truncated SVD（截斷奇異值分解）

```python
from sklearn.decomposition import TruncatedSVD

# SVD 降維
svd = TruncatedSVD(
    n_components=1024,  # 降至 1024 維
    random_state=42
)
reduced_features = svd.fit_transform(sparse_matrix)
```

**降維效果**：
- 原始維度: 138,493（使用者數）
- 降維後: 1024 維（99.3% 維度壓縮）
- 性能影響: +0.1%（略優於純 KNN）
- 計算加速: SVD 訓練 ~90s，但推理時特徵維度降低

### 2. 評估方法與優化策略

#### 評估協議

採用 **Leave-One-Out 交叉驗證**：

```python
# 每個使用者保留1個項目作為測試
for user_id in test_users:
    # 1. 隱藏一個評分項目
    test_item = user_ratings.pop()  
    
    # 2. 基於剩餘評分進行推薦
    recommendations = model.recommend(user_id, n=10)
    
    # 3. 檢查測試項目是否在推薦列表中
    hit = (test_item in recommendations)
```

**評估指標**：

- **Hit Rate@10**: 前 10 個推薦中是否命中測試項目（@10 表示推薦列表長度 N=10）
  $$\text{Hit Rate@10} = \frac{\text{命中使用者數}}{\text{總使用者數}}$$
  
- **NDCG@10**: 歸一化折扣累積增益，考慮排序位置（@10 表示僅計算前 10 個推薦）
  $$\text{NDCG@10} = \frac{DCG@10}{IDCG@10}$$
  
- **RMSE**: 評分預測誤差（均方根誤差）
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

> 💡 **重要區分**: 
> - **推薦列表長度（N=10）**: Hit Rate@10、NDCG@10 中的「@10」
> - **KNN 鄰居數量（K=5~50）**: k_neighbors 參數，影響推薦質量的關鍵超參數

#### 分層評估策略

本研究採用**分層抽樣評估**，平衡實驗效率與統計可靠性：

**階段一：快速網格搜索（235+ 配置）**
- **樣本規模**: 20,000 個使用者（14.4% 全使用者）
- **統計誤差**: 95% 置信區間 ±0.7%
- **執行時間**: ~2-3 分鐘/配置
- **用途**: 快速探索配置空間，識別最優區域

**為何選擇 20K 樣本？**

這個樣本規模是基於**統計學理論**和**工程實務**的最優平衡：

| 樣本規模 | 統計誤差 | 計算時間 | 記憶體需求 | 實用性 |
|---------|---------|---------|-----------|-------|
| 500 | ±4.4% | ~2 秒 | ~50 MB | 快速原型 |
| **20,000** | **±0.7%** ✅ | **~2 分鐘** ✅ | **~2.9 GB** ✅ | **最佳平衡** |
| 138,493 | ±0.3% | ~9 分鐘（理論） | **~98 GB** ⚠️ | 不實用 |

**學術依據**:
- 符合推薦系統領域標準（Cremonesi et al., 2010）
- 20K 樣本的誤差範圍（±0.7%）遠小於配置間差異（>1%）
- 相對性能排序在大樣本下具有高度穩定性

#### KNN 預計算優化策略

**核心問題**: KNN 是惰性學習算法（Lazy Learning），沒有實際的"訓練"階段，僅在 `fit()` 時存儲特徵矩陣。評估時每個使用者的 `predict()` 和 `recommend()` 都會調用 `kneighbors()`，導致重複計算。

**傳統流程（無快取）**：
```python
# 每次評估都重新計算鄰居
for user in test_users:  # 500 個使用者
    # predict() 調用 1 次 kneighbors()
    # recommend() 調用 1 次 kneighbors()
    # 共 2 次 × 500 = 1000 次 kneighbors() 呼叫
    # 每次計算與所有使用者的距離（brute-force）
    distances, indices = knn.kneighbors(user_vec, k+1)  # O(n × d)
    
# 總耗時：1800 秒（30 分鐘）⚠️
# 20K 使用者：估計需要 72000 秒（20 小時）⚠️⚠️⚠️
```

**優化方案：全域快取機制（🚀 1000x+ 加速）**

```python
# 來源: src/movie_recommendation/models.py

class KNNRecommender:
    def fit(self, features):
        """KNN 無訓練階段，僅存儲特徵並預計算鄰居"""
        # 1. 套用 sklearn KNN 的殼（無實際訓練）
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn.fit(features)  # 僅存儲特徵，無模型訓練
        
        # 2. 全域快取：一次性計算所有使用者的鄰居
        if n_users <= 20000:  # 記憶體閾值
            distances, indices = self.knn.kneighbors(features, k+1)
            self._neighbor_indices_all = indices[:, 1:]  # 排除自身
            self._neighbor_similarities_all = 1 - distances[:, 1:]
            # 成本: O(n² × d) 一次性
            # 評估時: O(1) 查表 ✅
    
    def find_neighbors(self, user_idx):
        """查找鄰居：優先使用快取"""
        if self._neighbor_indices_all is not None:
            # 直接從快取讀取（O(1)）
            return self._neighbor_indices_all[user_idx]
        else:
            # 退化到動態計算（O(n × d)）
            return self.knn.kneighbors(user_vec, k+1)
```

**實測性能對比**（500 個測試使用者，10M 資料）：

| 策略 | fit() 時間 | 評估時間 | 記憶體使用 | 總時間 | 加速比 |
|-----|----------|---------|-----------|--------|--------|
| **無快取** | ~0.05s | **1800s** (30分鐘) | ~300 MB | 1800s | 1x |
| **全域快取** ✅ | ~0.05s | **2s** ⚡ | ~2.8 GB | 2s | **🚀 1000x+** |

**實測性能對比**（20K 使用者完整評估，20M 資料，全域快取）：

| 配置 | fit() 時間 | 評估時間 | 記憶體使用 | Hit Rate |
|-----|----------|---------|-----------|----------|
| 純 KNN (K=5) | 0.06s | 65s | 2.8 GB | 62.18% |
| 純 KNN (K=35) | 0.06s | 120s | 2.8 GB | 67.67% |
| SVD+KNN (1024, K=40) | 0.06s | 119s | 4.5 GB | 67.74% |

> 註：SVD 降維時間為 90s，但這是特徵工程階段，不計入 KNN "訓練"

**關鍵技術點**：

1. **KNN 無訓練本質**：
   - `fit()` 僅存儲特徵矩陣，沒有參數學習
   - 實際計算發生在 `kneighbors()` 調用時
   - 每次 `predict()` 和 `recommend()` 都觸發距離計算

2. **重複計算問題**：
   - 傳統流程：每個使用者評估 2 次 kneighbors()
   - 500 使用者 × 2 次 = 1000 次重複計算
   - 每次計算與所有使用者的距離：O(n × d)

3. **快取優化效果**：
   - **🚀 1000x+ 評估加速**：1800s → 2s（500 使用者）
   - **🚀 1000x+ 評估加速**：30 分鐘 → 2 秒
   - **記憶體代價**：300 MB → 2.8 GB（可接受的權衡）
   - **可擴展性**：20K 使用者評估僅需 65-120 秒

4. **工程權衡**：
   - ✅ **≤20K 使用者**: 全域快取可行（<8 GB RAM）
   - ⚠️ **>20K 使用者**: 退化到批次計算或動態計算
   - 🎯 **本研究**: 20K 樣本 + 全域快取 = 最佳平衡
 **🚀 1000x+ 評估加速**，使得大規模網格搜索（235+ 配置）在消費級硬體上可行。若無此優化，235 個實驗需要 **~196 天**，優化後僅需 **~4.7 小時**
> 💡 **技術創新**: 將 KNN 的 lazy learning 特性轉化為優勢，透過一次性預計算實現千倍級評估加速，使得大規模網格搜索（235+ 配置）在消費級硬體上可行。

### 3. 記憶體與計算優化

**稀疏矩陣優化**：

```python
import scipy.sparse as sp

# 構建稀疏評分矩陣
rating_matrix = sp.csr_matrix(
    (ratings, (user_indices, item_indices)),
    shape=(n_users, n_items)
)

# 記憶體節省
dense_memory = n_users * n_items * 8  # float64
sparse_memory = len(ratings) * (8 + 4 + 4)  # value + row + col
compression_ratio = 1 - (sparse_memory / dense_memory)
# 壓縮率: ~99.5%（20M 評分 vs 3.7B 可能位置）
```

**效能提升**：
- **記憶體使用**: ~160 MB（稀疏） vs ~27 GB（稠密）
- **壓縮率**: 99.4%
- **計算加速**: 稀疏矩陣運算針對非零元素優化

**批次處理策略**：

```python
# 用於資料集分析（避免一次性載入全部資料）
chunk_size = 500000
for chunk in pd.read_csv('ratings.csv', chunksize=chunk_size):
    process_chunk(chunk)  # 增量更新統計量
```

**向量化計算**：

```python
# ❌ 慢速迴圈
for i in range(n_users):
    for j in range(k):
        score += similarity[i, j] * rating[j]

# ✅ 向量化（100x 加速）
scores = similarity @ ratings  # NumPy 矩陣乘法
```

---

## 🔍 深度分析：SVD + KNN vs 純 KNN

### 性能對比總結

| 維度 | SVD + KNN | 純 KNN | 差異 | 結論 |
|------|-----------|--------|------|------|
| **Hit Rate@10** | 67.74% | 67.67% | +0.07% | SVD 略優 |
| **NDCG@10** | 0.5363 | 0.5324 | +0.73% | SVD 排序更好 |
| **RMSE** | 0.9600 | 0.9507 | +0.98% | 純 KNN 預測更準 |
| **訓練時間** | ~90s | ~5s | +17x | SVD 成本高 |
| **推理速度** | 相同 | 相同 | - | 評估階段無差異 |

### 關鍵洞察

#### 1. SVD 的有限收益（+0.07%）

**實驗證據**：
- 最佳 SVD+KNN: SVD=1024, K=40 → 67.74%
- 最佳純 KNN: K=35 → 67.67%
- 性能差距僅 **0.07%**（統計誤差 ±0.7% 範圍內）

**可能原因**：
- ✅ MovieLens 資料質量高，噪音少
- ✅ KNN 本身已能有效捕捉協同訊號
- ✅ SVD 降維可能移除了部分有用的長尾訊號

**結論**: 對於 MovieLens 20M，SVD 降維**並非必要**。

#### 2. 高維度的邊際收益（SVD=1024 vs 512）

| SVD 維度 | 最佳 K | Hit Rate | 相對提升 |
|---------|-------|----------|---------|
| 512 | 35 | 67.54% | 基準 |
| 1024 | 40 | 67.74% | +0.30% |

**成本對比**：
- SVD=512: 訓練 ~45s，記憶體 ~1.5 GB
- SVD=1024: 訓練 ~90s，記憶體 ~2.9 GB

**結論**: SVD=1024 提升 0.3%，但成本倍增。**SVD=512 是更實用的選擇**。

#### 3. 最佳 K 值差異

- **SVD + KNN**: K=40 最優
- **純 KNN**: K=35 最優

**分析**:
- SVD 降維後特徵更平滑，可容忍更多鄰居（K=40）
- 純 KNN 在原始空間，K=35 已達到最佳平衡
- 過大 K 值（>45）對兩者都有害

#### 4. 實用建議

**場景一：追求極致性能（+0.07%）**
```python
config = {
    'use_svd': True,
    'n_components': 1024,
    'k_neighbors': 40
}
# Hit Rate: 67.74%
# 訓練時間: ~90s
```

**場景二：平衡性能與成本（推薦）**
```python
config = {
    'use_svd': True,
    'n_components': 512,
    'k_neighbors': 35
}
# Hit Rate: 67.54% (-0.3%)
# 訓練時間: ~45s (-50%)
```

**場景三：最簡配置（快速原型）**
```python
config = {
    'use_svd': False,
    'k_neighbors': 35
}
# Hit Rate: 67.67% (-0.1%)
# 訓練時間: ~5s (-95%)
```

**我們的選擇**: **場景一**（SVD=1024, K=40）

**理由**:
1. 性能最優（即使提升微小）
2. 訓練成本可接受（~90s 一次性成本）
3. 推理性能相同（評估階段無差異）
4. 完整性驗證（確認 SVD 極限效果）

---

## 🚀 快速開始

### 環境需求

- Python 3.12+
- 8GB+ RAM（20K 樣本評估需求）
- 推薦使用 `uv` 作為包管理器

### 安裝

```bash
# 克隆專案
git clone https://github.com/TsukiSama9292/1141_DataScience.git
cd 1141_DataScience

# 安裝依賴（使用 uv）
uv sync
```

### 執行實驗

```bash
# 方式1：執行所有實驗並生成報告
uv run python main.py

# 方式2：只執行特定階段
uv run python main.py --stage SVD_KNN_GRID

# 方式3：只生成報告（使用已完成的實驗）
uv run python main.py --report-only
```

### 查看結果

```bash
# 實驗日誌
ls log/*.json

# 報告和圖表
ls reports/figures/*.png

# 最佳配置摘要
cat reports/summary.md
```

---

## 📚 項目結構

```
1141_DataScience/
├── README.md                          # 本檔案
├── main.py                            # 主執行入口
├── pyproject.toml                     # 依賴配置（uv）
│
├── configs/
│   └── experiments.json               # 實驗配置（235+ 個實驗）
│
├── src/movie_recommendation/          # 核心代碼
│   ├── data_loader.py                 # 資料載入（MovieLens 20M）
│   ├── feature_engineering.py         # 特徵工程（SVD 降維）
│   ├── models.py                      # KNN 推薦模型（含預計算優化）
│   ├── evaluation.py                  # 評估指標（Hit Rate, NDCG, RMSE）
│   ├── experiment.py                  # 實驗執行器
│   ├── experiment_runner.py           # 多階段實驗管理
│   ├── config_loader.py               # 配置載入器
│   ├── analysis.py                    # 結果分析
│   ├── report_generator.py            # 報告生成（含熱力圖）
│   └── utils.py                       # 工具函數
│
├── log/                               # 實驗結果（JSON）
│   ├── FILTER_001.json ~ 006.json     # 過濾階段（6個）
│   ├── KNN_BASELINE_001.json ~ 010    # 純 KNN（10個）
│   ├── SVD_KNN_GRID_001.json ~ 100    # 網格搜索（100個）
│   ├── SVD_KNN_EXPAND_001.json ~ 117  # 擴展搜索（117個）
│   ├── BIAS_001.json ~ 002.json       # 偏差測試（2個）
│   └── OPT_001.json ~ 002.json        # 優化測試（2個）
│
├── reports/                           # 報告與可視化
│   ├── summary.md                     # 實驗摘要表格
│   ├── dataset_statistics_full.json   # 完整資料集統計（20M 評分）
│   └── figures/                       # 可視化圖表
│       ├── svd_knn_grid_heatmap.png       # SVD×KNN 網格熱力圖 ⭐
│       ├── svd_knn_expand_heatmap.png     # 擴展搜索熱力圖
│       ├── svd_dimension_analysis.png     # SVD 維度分析
│       ├── knn_k_value_analysis.png       # KNN K 值分析
│       ├── stage_comparison.png           # 階段對比
│       ├── data_rating_distribution_full.png  # 評分分布
│       ├── data_user_activity_long_tail_full.png  # 使用者活躍度
│       └── data_movie_popularity_long_tail_full.png  # 電影流行度
│
└── tools/                             # 工具腳本
    ├── generate_grid_config.py        # 網格配置生成器
    ├── expand_low_dim_high_knn.py     # 擴展實驗生成器
    └── ...
```

---

## 🎯 關鍵發現與建議

### ✅ 有效策略

1. **高維度 SVD（1024）**
   - ✅ 達到最優性能（+0.07% vs 純 KNN）
   - ✅ 特徵平滑化，降低噪音
   - ⚠️ 訓練成本較高（~90s）

2. **適中 K 值（35-40）**
   - ✅ 性能最優範圍
   - ✅ 避免過度擬合（K>45 性能下降）
   - ✅ 穩健選擇

3. **純 KNN 基準（K=35）**
   - ✅ 性能優異（67.67%，僅次 SVD+KNN 0.07%）
   - ✅ 訓練極快（~5s）
   - ✅ 實用首選

### ❌ 無效策略

1. **時間衰減（Time Decay）**
   - ❌ Hit Rate 下降至 66.14%（-2.4%）
   - 原因：電影偏好相對穩定，經典電影持續受歡迎
   - 結論：**不適用**於電影推薦場景

2. **偏差校正（Item Bias）**
   - ❌ 無性能提升（67.74% 維持）
   - 原因：熱門電影確實值得推薦
   - 結論：**不需要**過度校正

3. **極低/極高 K 值**
   - ❌ K<15 或 K>45 性能顯著下降
   - 原因：欠擬合或過度擬合
   - 結論：K=25-40 為最佳範圍

### 🎯 生產環境推薦配置

**場景1：追求極致性能**
```python
config = {
    'use_svd': True,
    'n_components': 1024,
    'k_neighbors': 40,
    'min_item_ratings': 0,
    'use_item_bias': False,
    'use_time_decay': False
}
# 預期性能: Hit Rate@10 = 67.74%
# 訓練時間: ~90s（一次性）
# 評估時間: ~2 分鐘（20K 使用者）
```

**場景2：平衡性能與成本（推薦）**
```python
config = {
    'use_svd': True,
    'n_components': 512,
    'k_neighbors': 35,
    'min_item_ratings': 0
}
# 預期性能: Hit Rate@10 = 67.54% (-0.3%)
# 訓練時間: ~45s (-50%)
```

**場景3：快速原型與實驗**
```python
config = {
    'use_svd': False,
    'k_neighbors': 35,
    'min_item_ratings': 0
}
# 預期性能: Hit Rate@10 = 67.67% (-0.1%)
# 訓練時間: ~5s (-95%)
```

---

## 📖 學術參考

- **協同過濾**: Sarwar, B., et al. (2001). *Item-based collaborative filtering recommendation algorithms*. WWW.
- **評估方法**: Cremonesi, P., et al. (2010). *Performance of recommender algorithms on top-n recommendation tasks*. RecSys.
- **資料集**: Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets*. ACM TIST.
- **SVD 應用**: Koren, Y., et al. (2009). *Matrix factorization techniques for recommender systems*. Computer.

---

## 📄 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 檔案

---

## 🙏 致謝

- **資料來源**: [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) by GroupLens Research
- **開發工具**: Python, Scikit-learn, NumPy, Pandas, Matplotlib
- **包管理**: uv (快速、可靠的 Python 包管理器)