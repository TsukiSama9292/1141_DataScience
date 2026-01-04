#!/usr/bin/env python3
"""
生成低維度 SVD + 高 KNN 擴展實驗配置

用於填補研究空白：SVD 維度 ≤1024，KNN 鄰居 40~80
"""

import json
import sys
from pathlib import Path

# 添加 src 到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.generate_grid_config import (
    update_config_with_grid,
    preview_grid
)


def main():
    """主函數"""
    config_path = Path(__file__).parent.parent / 'configs' / 'experiments.json'
    
    # 定義低維度 SVD（2~512）和高 KNN（40~80，每5為一個步長）
    svd_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    knn_values = [40, 45, 50, 55, 60, 65, 70, 75, 80]
    
    print("\n" + "="*80)
    print("🔬 生成低維度 SVD + 高 KNN 擴展實驗配置")
    print("="*80)
    print("\n📊 實驗範圍:")
    print(f"   SVD 維度: {svd_values}")
    print(f"   KNN 鄰居: {knn_values}")
    print(f"   總實驗數: {len(svd_values) * len(knn_values)} (9×9=81)")
    
    print("\n💡 目的:")
    print("   - 填補研究空白：之前只測試了 SVD≥1024 的高 KNN 情況")
    print("   - 探索低維度 SVD 與高 KNN 的配合效果")
    print("   - 檢驗是否存在低維度高 KNN 的最優解")
    
    # 預覽
    print("\n" + "="*80)
    print("📋 實驗預覽")
    print("="*80)
    preview_grid(svd_values, knn_values)
    
    # 確認
    print("\n" + "="*80)
    response = input("\n是否將這些實驗添加到 SVD_KNN_EXPAND 階段？(y/N): ")
    
    if response.lower() == 'y':
        print("\n🚀 開始更新配置...")
        
        try:
            update_config_with_grid(
                config_path=config_path,
                svd_values=svd_values,
                knn_values=knn_values,
                remove_old_stages=False,  # 不移除任何舊階段
                stage_id='SVD_KNN_EXPAND',  # 添加到 EXPAND 階段
                insert_after='SVD_KNN_GRID',  # 在 GRID 之後
                append_mode=True,  # 使用附加模式
                skip_existing=True  # 跳過已存在的實驗
            )
            
            print("\n" + "="*80)
            print("✅ 配置更新完成！")
            print("="*80)
            print("\n📝 下一步:")
            print("   1. 檢查 configs/experiments.json，確認新實驗已添加")
            print("   2. 啟用 SVD_KNN_EXPAND 階段: 設置 'enabled': true")
            print("   3. 執行實驗: python main.py --stage SVD_KNN_EXPAND")
            print()
            
        except Exception as e:
            print(f"\n❌ 更新失敗: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("\n❌ 已取消")
        print("💡 提示: 你可以稍後再次運行此腳本")


if __name__ == '__main__':
    main()
