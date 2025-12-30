"""
# 基於最近鄰演算法的協同過濾推薦系統
大家好！在這個 Python 程式中，將建立一個電影推薦系統。
在這個系統中，將使用最近鄰演算法和協同過濾。
資料集使用的是 MovieLens 20M 資料集。
祝您機器學習愉快！

### **讓我​​們來描述一下資料集：**

此資料集包含 6 個 CSV 檔案。我使用了 movie.csv 和 rating.csv 這兩個檔案。讓我們來分析這些 CSV 檔案。

rating.csv 檔案包含使用者對電影的評分：
* userId
* movieId
* rating
* timestamp

movie.csv 檔案包含電影資訊：
* movieId
* title
* genres

### 

"""
# 載入套件
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import kagglehub

# 下載資料集
path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")

# 載入資料
data_movie = pd.read_csv(f"{path}/movie.csv")
data_rating = pd.read_csv(f"{path}/rating.csv")

# 在 movie.csv 檔案中，我們將使用 movieid 和 title 欄位。
# 我們將使用這兩列來建立一個新的資料框。
# 同時，在 rating.csv 檔案中，我們將使用 userId、movieid 和 rating 欄位。我們將使用這三列來建立一個新的資料框。

movie = data_movie.loc[:, ["movieId", "title"]]
rating = data_rating.loc[:, ["userId", "movieId", "rating"]]

# 我們將兩個資料框（影片和評分）合併，並創建電影與使用者矩陣。

data = pd.merge(movie,rating)
data = data.iloc[:1000000,:]
user_movie_table = data.pivot_table(index = ["title"],columns = ["userId"],values = "rating").fillna(0)
# user_movie_table.head(10) # 顯示前 10 筆資料 --- 在終端機忽略 --- 

"""
# 什麼是推薦系統？
* 推薦系統是基於使用者過去的行為，預測使用者對某個項目或物品的偏好程度。
* 例如 Netflix 就使用了推薦系統。它會根據用戶過去的觀影和投票等活動，向他們推薦新電影。
* 推薦系統的目的是向使用者推薦他們以前從未接觸過的新事物。
* 推薦系統有多種方法。在本範例中，我使用了協同過濾方法。
## 協同過濾
協同過濾結合使用者自身和其他使用者的經驗進行推薦。協同過濾有兩種方法：基於使用者的協同過濾和基於物品的協同過濾。
### 基於使用者的協同過濾
它計算使用者與物品在使用者-物品矩陣中的相似度。例如，假設有兩個使用者。第一個用戶觀看了《魔戒》和《哈比人》兩部電影。第二個用戶只看了《魔戒》。系統會向第二個使用者推薦《哈比人》。
基於使用者的協同過濾存在一些問題。在這個系統中，矩陣的每一行代表一個使用者。因此，比較使用者之間的相似度運算量龐大，耗費大量的運算資源。此外，人們的習慣會不斷改變。因此，隨著時間的推移，做出正確且有用的推薦可能變得困難。
為了解決這些問題，讓我們來看看另一個推薦系統：基於物品的協同過濾。
### 基於物品的協同過濾
它計算使用者與物品矩陣中物品之間的相似度。例如，假設有兩部電影：《魔戒》和《哈比人》。三個人分別觀看了《魔戒》和《哈比人》。如果第四個人也看了《魔戒》，他/她可能也會喜歡《哈比人》。因此，系統會向第四個人推薦《哈比人》。
通常，推薦系統使用基於物品的協同過濾。基於物品的協同過濾改善了基於使用者的協同過濾，解決了使用者協同過濾的問題。因為人們的想法和習慣會改變，而物品不會改變，所以基於物品的協同過濾更受歡迎。
"""

# 選擇隨機電影進行推薦
query_index = np.random.choice(user_movie_table.shape[0])
print("Choosen Movie is: ",user_movie_table.index[query_index])

"""
有很多方法可以找到相似度。在本範例中，我使用了 K 近鄰演算法來找到相似的電影。
## 什麼是K近鄰演算法？
K近鄰演算法既可以用於分類問題，也可以用於迴歸問題。在分類問題中，為了預測一個實例的標籤，我們首先基於距離度量找到與給定實例最接近的k個實例，然後基於多數投票機製或加權多數投票機制（距離越近的鄰居權重越高）來預測標籤。
K近鄰演算法基於給定的距離度量（例如歐氏距離、Jaccard相似度、閔可夫斯基距離或自訂距離度量）找到與特定實例最相似的k個項目。在我的模型中，我使用了餘弦距離作為度量。
"""
# 將 user_movie_table 轉換為稀疏矩陣
user_movie_table_matrix = csr_matrix(user_movie_table.values)
# 建立 KNN 模型
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
# 訓練模型
model_knn.fit(user_movie_table_matrix)
# 找到相似的電影，這裡我們找前 6 個鄰居（包含自己）
distances, indices = model_knn.kneighbors(user_movie_table.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 6)

# 在以下部分，您可以查看有關電影《親密接觸》的推薦。

movie = []
distance = []

for i in range(0, len(distances.flatten())):
    if i != 0:
        movie.append(user_movie_table.index[indices.flatten()[i]])
        distance.append(distances.flatten()[i])

m=pd.Series(movie,name='movie')
d=pd.Series(distance,name='distance')
recommend = pd.concat([m,d], axis=1)
recommend = recommend.sort_values('distance',ascending=False)

print('Recommendations for {0}:\n'.format(user_movie_table.index[query_index]))
for i in range(0,recommend.shape[0]):
    print('{0}: {1}, with distance of {2}'.format(i, recommend["movie"].iloc[i], recommend["distance"].iloc[i]))