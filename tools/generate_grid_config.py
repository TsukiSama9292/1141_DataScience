#!/usr/bin/env python3
"""
網格搜索配置生成器

自動生成 SVD × KNN 的笛卡爾積配置
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def generate_grid_experiments(
    svd_values: List[int],
    knn_values: List[int],
    stage_id: str = "SVD_KNN_GRID",
    base_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    生成網格搜索實驗配置
    
    Args:
        svd_values: SVD 維度列表
        knn_values: KNN 鄰居數列表
        stage_id: 階段 ID
        base_config: 基礎配置
    
    Returns:
        完整的階段配置
    """
    if base_config is None:
        base_config = {
            "data_limit": None,
            "min_item_ratings": 0,
            "use_svd": True
        }
    
    experiments = []
    exp_counter = 1
    
    # 生成所有 SVD × KNN 組合
    for svd in svd_values:
        for knn in knn_values:
            exp_id = f"{stage_id}_{exp_counter:03d}"
            exp_name = f"SVD={svd}×KNN={knn}"
            
            # 計算 2 的冪次表示
            svd_power = svd.bit_length() - 1 if svd > 0 else 0
            knn_power = knn.bit_length() - 1 if knn > 0 else 0
            description = f"2^{svd_power}維度 × 2^{knn_power}鄰居"
            
            experiment = {
                "id": exp_id,
                "name": exp_name,
                "description": description,
                "config": {
                    "n_components": svd,
                    "k_neighbors": knn
                }
            }
            
            experiments.append(experiment)
            exp_counter += 1
    
    stage_config = {
        "name": "SVD×KNN 網格搜索",
        "description": "同時測試所有 SVD 和 KNN 組合，找出最佳配對",
        "enabled": True,
        "base_config": base_config,
        "experiments": experiments
    }
    
    return stage_config


def update_config_with_grid(
    config_path: Path,
    svd_values: List[int] = None,
    knn_values: List[int] = None,
    remove_old_stages: bool = True
):
    """
    更新配置檔案，添加網格搜索階段
    
    Args:
        config_path: 配置檔案路徑
        svd_values: SVD 維度列表
        knn_values: KNN 鄰居數列表
        remove_old_stages: 是否移除舊的 SVD_COARSE 和 KNN_COARSE 階段
    """
    # 預設值：SVD 使用 2^n (n=1..10)，KNN 使用 5*n (n=1..10)
    if svd_values is None:
        svd_values = [2**n for n in range(1, 11)]  # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    if knn_values is None:
        knn_values = [5*n for n in range(1, 11)]   # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # 載入現有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 生成網格搜索配置
    grid_stage = generate_grid_experiments(svd_values, knn_values)
    
    # 移除舊階段
    if remove_old_stages:
        stages_to_remove = ['SVD_COARSE', 'KNN_COARSE', 'SVD_FINE', 'KNN_FINE']
        for stage in stages_to_remove:
            if stage in config['stages']:
                del config['stages'][stage]
                print(f"✓ 已移除階段: {stage}")
    
    # 添加或更新網格搜索階段（在 FILTER 之後）
    new_stages = {}
    grid_added = False
    for key, value in config['stages'].items():
        if key == 'SVD_KNN_GRID':
            # 跳過舊的 SVD_KNN_GRID，稍後會添加新的
            continue
        new_stages[key] = value
        if key == 'FILTER':
            new_stages['SVD_KNN_GRID'] = grid_stage
            grid_added = True
            print(f"✓ 已添加/更新階段: SVD_KNN_GRID（{len(grid_stage['experiments'])} 個實驗）")
    
    # 如果沒有 FILTER 階段，直接添加到最後
    if not grid_added:
        new_stages['SVD_KNN_GRID'] = grid_stage
        print(f"✓ 已添加階段: SVD_KNN_GRID（{len(grid_stage['experiments'])} 個實驗）")
    
    config['stages'] = new_stages
    
    # 更新 metadata
    config['metadata']['strategy'] = 'grid_search_with_power_of_2'
    config['metadata']['description'] = '電影推薦系統實驗配置檔案 - 使用網格搜索找出最佳 SVD×KNN 配對'
    
    # 儲存更新後的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 配置更新完成！")
    print(f"📊 網格搜索: {len(svd_values)} SVD × {len(knn_values)} KNN = {len(grid_stage['experiments'])} 個實驗")


def preview_grid(svd_values: List[int], knn_values: List[int]):
    """預覽網格搜索配置"""
    print("\n" + "="*80)
    print("📋 網格搜索配置預覽")
    print("="*80)
    
    print(f"\nSVD 維度: {svd_values}")
    print(f"KNN 鄰居: {knn_values}")
    print(f"總實驗數: {len(svd_values) * len(knn_values)}")
    
    print("\n實驗列表:")
    counter = 1
    for svd in svd_values:
        for knn in knn_values:
            print(f"  {counter:2d}. SVD={svd:3d} × KNN={knn:2d}")
            counter += 1
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='網格搜索配置生成器')
    parser.add_argument(
        '--preview',
        action='store_true',
        help='只預覽，不修改配置檔案'
    )
    parser.add_argument(
        '--svd',
        type=int,
        nargs='+',
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help='SVD 維度列表（預設: 2^n, n=1..10, 共10個值）'
    )
    parser.add_argument(
        '--knn',
        type=int,
        nargs='+',
        default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='KNN 鄰居數列表（預設: 5*n, n=1..10, 共10個值）'
    )
    parser.add_argument(
        '--keep-old',
        action='store_true',
        help='保留舊的 SVD_COARSE 和 KNN_COARSE 階段'
    )
    
    args = parser.parse_args()
    
    if args.preview:
        preview_grid(args.svd, args.knn)
    else:
        config_path = Path(__file__).parent.parent / 'configs' / 'experiments.json'
        
        print("\n" + "="*80)
        print("🔧 更新配置檔案")
        print("="*80)
        print(f"\n配置檔案: {config_path}")
        
        # 預覽
        preview_grid(args.svd, args.knn)
        
        # 確認
        response = input("\n是否繼續更新配置檔案？(y/N): ")
        if response.lower() == 'y':
            update_config_with_grid(
                config_path,
                args.svd,
                args.knn,
                remove_old_stages=not args.keep_old
            )
        else:
            print("❌ 已取消")
