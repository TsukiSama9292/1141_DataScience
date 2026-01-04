#!/usr/bin/env python3
"""
網格搜索配置生成器

自動生成 SVD × KNN 的笛卡爾積配置

新功能：
- 支援檢測現有實驗並避免重複
- 可以添加新實驗到現有階段
- 自動分配實驗ID
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Set, Tuple


def get_existing_experiments(stage_config: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """
    獲取現有階段中已存在的實驗配置 (SVD, KNN) 組合
    
    Args:
        stage_config: 階段配置
    
    Returns:
        已存在的 (n_components, k_neighbors) 組合集合
    """
    existing = set()
    for exp in stage_config.get('experiments', []):
        config = exp.get('config', {})
        n_comp = config.get('n_components')
        k_neigh = config.get('k_neighbors')
        if n_comp is not None and k_neigh is not None:
            existing.add((n_comp, k_neigh))
    return existing


def get_next_experiment_id(stage_config: Dict[str, Any], stage_id: str) -> int:
    """
    獲取下一個可用的實驗ID編號
    
    Args:
        stage_config: 階段配置
        stage_id: 階段ID
    
    Returns:
        下一個可用的編號
    """
    existing_ids = []
    for exp in stage_config.get('experiments', []):
        exp_id = exp.get('id', '')
        if exp_id.startswith(f"{stage_id}_"):
            try:
                num = int(exp_id.split('_')[-1])
                existing_ids.append(num)
            except ValueError:
                pass
    
    return max(existing_ids, default=0) + 1


def generate_grid_experiments(
    svd_values: List[int],
    knn_values: List[int],
    stage_id: str = "SVD_KNN_GRID",
    base_config: Dict[str, Any] = None,
    existing_stage_config: Dict[str, Any] = None,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    生成網格搜索實驗配置
    
    Args:
        svd_values: SVD 維度列表
        knn_values: KNN 鄰居數列表
        stage_id: 階段 ID
        base_config: 基礎配置
        existing_stage_config: 現有的階段配置（用於檢測重複）
        skip_existing: 是否跳過已存在的實驗
    
    Returns:
        完整的階段配置
    """
    if base_config is None:
        base_config = {
            "data_limit": None,
            "min_item_ratings": 0,
            "use_svd": True
        }
    
    # 獲取已存在的實驗
    existing_experiments = set()
    exp_counter = 1
    
    if existing_stage_config:
        existing_experiments = get_existing_experiments(existing_stage_config)
        exp_counter = get_next_experiment_id(existing_stage_config, stage_id)
        print(f"\nℹ️  檢測到 {len(existing_experiments)} 個已存在的實驗")
        print(f"ℹ️  下一個實驗ID將從 {stage_id}_{exp_counter:03d} 開始")
    
    experiments = []
    skipped_count = 0
    
    # 生成所有 SVD × KNN 組合
    for svd in svd_values:
        for knn in knn_values:
            # 檢查是否已存在
            if skip_existing and (svd, knn) in existing_experiments:
                skipped_count += 1
                continue
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
    
    if skip_existing and skipped_count > 0:
        print(f"✅ 跳過 {skipped_count} 個已存在的實驗")
        print(f"➕ 將添加 {len(experiments)} 個新實驗")
    
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
    remove_old_stages: bool = True,
    stage_id: str = "SVD_KNN_GRID",
    insert_after: str = "FILTER",
    append_mode: bool = False,
    skip_existing: bool = True
):
    """
    更新配置檔案，添加網格搜索階段
    
    Args:
        config_path: 配置檔案路徑
        svd_values: SVD 維度列表
        knn_values: KNN 鄰居數列表
        remove_old_stages: 是否移除舊的 SVD_COARSE 和 KNN_COARSE 階段
        stage_id: 階段 ID（預設: SVD_KNN_GRID）
        insert_after: 在此階段之後插入（預設: FILTER）
        append_mode: 是否以附加模式（將新實驗添加到現有階段）
        skip_existing: 是否跳過已存在的實驗（僅在 append_mode=True 時有效）
    """
    # 預設值：SVD 使用 2^n (n=1..10)，KNN 使用 5*n (n=1..10)
    if svd_values is None:
        svd_values = [2**n for n in range(1, 11)]  # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    if knn_values is None:
        knn_values = [5*n for n in range(1, 11)]   # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # 載入現有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 獲取現有的階段配置（用於附加模式）
    existing_stage_config = None
    if append_mode and stage_id in config.get('stages', {}):
        existing_stage_config = config['stages'][stage_id]
        print(f"\nℹ️  附加模式：將新實驗添加到現有階段 {stage_id}")
    
    # 生成網格搜索配置
    grid_stage = generate_grid_experiments(
        svd_values, 
        knn_values, 
        stage_id=stage_id,
        existing_stage_config=existing_stage_config,
        skip_existing=skip_existing
    )
    
    # 如果是附加模式，合併現有和新的實驗
    if append_mode and existing_stage_config:
        # 保留現有階段的元資訊
        if 'name' in existing_stage_config:
            grid_stage['name'] = existing_stage_config['name']
        if 'description' in existing_stage_config:
            grid_stage['description'] = existing_stage_config['description']
        if 'enabled' in existing_stage_config:
            grid_stage['enabled'] = existing_stage_config['enabled']
        
        # 合併實驗列表
        existing_experiments = existing_stage_config.get('experiments', [])
        grid_stage['experiments'] = existing_experiments + grid_stage['experiments']
        
        print(f"\n✅ 合併完成：{len(existing_experiments)} 個現有 + {len(grid_stage['experiments']) - len(existing_experiments)} 個新實驗")
    
    # 移除舊階段
    if remove_old_stages:
        
        print(f"\n✅ 合併完成：{len(existing_experiments)} 個現有 + {len(grid_stage['experiments']) - len(existing_experiments)} 個新實驗")
    
    # 移除舊階段
    if remove_old_stages:
        stages_to_remove = ['SVD_COARSE', 'KNN_COARSE', 'SVD_FINE', 'KNN_FINE']
        for stage in stages_to_remove:
            if stage in config['stages']:
                del config['stages'][stage]
                print(f"✓ 已移除階段: {stage}")
    
    # 添加或更新網格搜索階段
    new_stages = {}
    grid_added = False
    for key, value in config['stages'].items():
        if key == stage_id:
            # 跳過舊的同名階段，稍後會添加新的
            continue
        new_stages[key] = value
        if key == insert_after:
            new_stages[stage_id] = grid_stage
            grid_added = True
            print(f"✓ 已添加/更新階段: {stage_id}（{len(grid_stage['experiments'])} 個實驗）")
    
    # 如果沒有指定的插入位置，直接添加到最後
    if not grid_added:
        new_stages[stage_id] = grid_stage
        print(f"✓ 已添加階段: {stage_id}（{len(grid_stage['experiments'])} 個實驗）")
    
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
    parser.add_argument(
        '--stage-id',
        type=str,
        default='SVD_KNN_GRID',
        help='階段 ID（預設: SVD_KNN_GRID）'
    )
    parser.add_argument(
        '--insert-after',
        type=str,
        default='FILTER',
        help='在此階段之後插入（預設: FILTER）'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='附加模式：將新實驗添加到現有階段，而不是取代'
    )
    parser.add_argument(
        '--include-existing',
        action='store_true',
        help='即使實驗已存在也重新生成（僅在 --append 模式下有效）'
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
                remove_old_stages=not args.keep_old,
                stage_id=args.stage_id,
                insert_after=args.insert_after,
                append_mode=args.append,
                skip_existing=not args.include_existing
            )
        else:
            print("❌ 已取消")
