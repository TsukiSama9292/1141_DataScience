"""
實驗執行器模組

負責從配置文件載入實驗並執行，支援自動級聯最佳配置。
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .config_loader import ConfigLoader, ExperimentSpec
from .experiment import Experiment, ExperimentConfig
from .utils import setup_logging

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """實驗執行器"""
    
    def __init__(
        self, 
        config_path: Optional[Path] = None,
        log_dir: Path = Path('log')
    ):
        """
        初始化實驗執行器
        
        Args:
            config_path: 配置文件路徑
            log_dir: 日誌目錄
        """
        self.config_loader = ConfigLoader(config_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 存儲最佳配置（用於級聯）
        self.best_configs = {}
    
    def is_completed(self, experiment_id: str, expected_config: Optional[Dict] = None) -> bool:
        """檢查實驗是否已完成且配置匹配
        
        Args:
            experiment_id: 實驗ID
            expected_config: 預期的配置參數（用於驗證配置一致性）
        
        Returns:
            True 如果實驗已完成且配置匹配（或不檢查配置）
        """
        json_path = self.log_dir / f"{experiment_id}.json"
        
        if not json_path.exists():
            return False
        
        # 如果提供了預期配置，需要驗證配置是否匹配
        if expected_config is not None:
            try:
                result = self.load_experiment_result(experiment_id)
                if result is None:
                    return False
                
                saved_config = result.get('config', {})
                
                # 檢查關鍵配置參數是否匹配
                # 只檢查 expected_config 中指定的參數
                for key, expected_value in expected_config.items():
                    saved_value = saved_config.get(key)
                    if saved_value != expected_value:
                        logger.info(f"⚠️  實驗 {experiment_id} 配置不匹配: {key}={saved_value} (期望 {expected_value})")
                        return False
                
                return True
                
            except Exception as e:
                logger.warning(f"⚠️  無法驗證實驗 {experiment_id} 的配置: {e}")
                return False
        
        # 如果沒有提供預期配置，只檢查文件存在性
        return True
    
    def load_experiment_result(self, experiment_id: str) -> Optional[Dict]:
        """載入實驗結果"""
        json_path = self.log_dir / f"{experiment_id}.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def run_experiment(
        self, 
        experiment_spec: ExperimentSpec,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        執行單個實驗
        
        Args:
            experiment_spec: 實驗規格
            force: 是否強制重新執行（即使已完成）
        
        Returns:
            實驗結果
        """
        experiment_id = experiment_spec.id
        
        # 檢查是否已完成（驗證配置一致性）
        # 提取關鍵配置參數用於驗證
        key_params = {}
        for key in ['n_components', 'k_neighbors', 'use_svd', 'min_item_ratings', 'data_limit']:
            if key in experiment_spec.config:
                key_params[key] = experiment_spec.config[key]
        
        if not force and self.is_completed(experiment_id, key_params):
            logger.info(f"⏭️  跳過實驗 {experiment_id}（已完成且配置匹配）")
            return {'status': 'skipped', 'reason': 'already_completed'}
        
        logger.info(f"🚀 開始實驗: {experiment_id} - {experiment_spec.name}")
        logger.info(f"   描述: {experiment_spec.description}")
        
        # 創建實驗配置
        config = ExperimentConfig(**experiment_spec.config)
        
        # 執行實驗
        try:
            start_time = time.time()
            experiment = Experiment(config, config_name=experiment_id)
            results = experiment.run()
            elapsed_time = time.time() - start_time
            
            logger.info(f"✅ 實驗完成: {experiment_id} (耗時: {elapsed_time:.1f}秒)")
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'elapsed_time': elapsed_time,
                'results': results
            }
        
        except Exception as e:
            logger.error(f"❌ 實驗失敗: {experiment_id}")
            logger.error(f"   錯誤: {str(e)}", exc_info=True)
            
            return {
                'status': 'failed',
                'experiment_id': experiment_id,
                'error': str(e)
            }
    
    def run_stage(
        self,
        stage: str,
        force: bool = False,
        cascade_best: bool = True
    ) -> Dict[str, Any]:
        """
        執行某個階段的所有實驗
        
        Args:
            stage: 階段名稱
            force: 是否強制重新執行
            cascade_best: 是否在階段完成後級聯最佳配置
        
        Returns:
            階段執行結果
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 執行階段: {stage}")
        logger.info(f"{'='*80}\n")
        
        experiments = self.config_loader.get_experiments(stage=stage)
        
        if not experiments:
            logger.warning(f"⚠️  階段 {stage} 沒有啟用的實驗")
            return {'status': 'no_experiments', 'stage': stage}
        
        results = []
        completed = 0
        skipped = 0
        failed = 0
        
        for exp_spec in experiments:
            result = self.run_experiment(exp_spec, force=force)
            results.append(result)
            
            if result['status'] == 'success':
                completed += 1
            elif result['status'] == 'skipped':
                skipped += 1
            elif result['status'] == 'failed':
                failed += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 階段 {stage} 完成")
        logger.info(f"{'='*80}")
        logger.info(f"✅ 成功: {completed}")
        logger.info(f"⏭️  跳過: {skipped}")
        logger.info(f"❌ 失敗: {failed}\n")
        
        # 如果启用级联，分析最佳配置并更新后续阶段
        if cascade_best and completed > 0:
            self._cascade_best_config(stage)
        
        return {
            'status': 'completed',
            'stage': stage,
            'total': len(experiments),
            'completed': completed,
            'skipped': skipped,
            'failed': failed,
            'results': results
        }
    
    def _cascade_best_config(self, completed_stage: str):
        """
        分析已完成階段的最佳配置，並級聯到後續階段
        
        Args:
            completed_stage: 已完成的階段名稱
        """
        logger.info(f"🔍 分析 {completed_stage} 階段的最佳配置...")
        
        # 載入該階段所有實驗結果
        experiments = self.config_loader.get_experiments(stage=completed_stage)
        best_exp = None
        best_hit_rate = -1
        
        for exp_spec in experiments:
            result = self.load_experiment_result(exp_spec.id)
            if result and 'metrics' in result:
                hit_rate = result['metrics'].get('hit_rate', 0)  # 修正：使用 'hit_rate' 而非 'hit_rate@10'
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_exp = exp_spec
        
        if best_exp:
            logger.info(f"🏆 最佳實驗: {best_exp.id} (Hit Rate@10 = {best_hit_rate:.3f})")
            
            # 提取關鍵配置參數
            config_updates = {}
            
            if completed_stage == 'SVD_KNN_GRID':
                # SVD_KNN_GRID階段：同時提取最佳SVD和KNN配置
                if best_exp.config.get('use_svd'):
                    config_updates['use_svd'] = True
                    config_updates['n_components'] = best_exp.config.get('n_components')
                    logger.info(f"   → SVD: n_components={config_updates['n_components']}")
                config_updates['k_neighbors'] = best_exp.config.get('k_neighbors')
                logger.info(f"   → KNN: k_neighbors={config_updates['k_neighbors']}")
            
            elif completed_stage.startswith('SVD'):
                # SVD階段：提取最佳SVD配置
                if best_exp.config.get('use_svd'):
                    config_updates['use_svd'] = True
                    config_updates['n_components'] = best_exp.config.get('n_components')
                    logger.info(f"   → SVD: n_components={config_updates['n_components']}")
            
            elif completed_stage == 'KNN_BASELINE':
                # KNN_BASELINE階段：不級聯，這是純KNN基準測試
                # 用於與 SVD+KNN 對比，不應影響後續階段
                logger.info(f"   → KNN Baseline: k_neighbors={best_exp.config.get('k_neighbors')} (不級聯)")
                return  # 直接返回，不更新後續階段
            
            elif completed_stage.startswith('KNN'):
                # 其他KNN階段：提取最佳KNN配置
                config_updates['k_neighbors'] = best_exp.config.get('k_neighbors')
                logger.info(f"   → KNN: k_neighbors={config_updates['k_neighbors']}")
            
            elif completed_stage == 'FILTER':
                # FILTER階段：不級聯，因為過濾會改變數據分佈
                # 這是數據預處理選項，應該獨立測試
                logger.info(f"   → Filter: min_item_ratings={best_exp.config.get('min_item_ratings', 0)} (不級聯)")
                return  # 直接返回，不更新後續階段
            
            elif completed_stage == 'BIAS':
                # BIAS階段：提取最佳偏差配置
                config_updates['use_item_bias'] = best_exp.config.get('use_item_bias', False)
                logger.info(f"   → Bias: use_item_bias={config_updates['use_item_bias']}")
            
            elif completed_stage == 'OPT':
                # OPT階段：提取最佳優化配置
                config_updates['use_time_decay'] = best_exp.config.get('use_time_decay', False)
                config_updates['half_life_days'] = best_exp.config.get('half_life_days', 500)
                config_updates['use_tfidf'] = best_exp.config.get('use_tfidf', False)
                logger.info(f"   → Optimization: use_time_decay={config_updates['use_time_decay']}, "
                           f"half_life_days={config_updates['half_life_days']}, "
                           f"use_tfidf={config_updates['use_tfidf']}")
            
            # 注意：不再處理 DS 階段，因為 data_limit 不應該被級聯
            
            # 存儲最佳配置
            self.best_configs[completed_stage] = config_updates
            
            # 更新後續階段的基礎配置
            self._update_subsequent_stages(completed_stage, config_updates)
    
    def _update_subsequent_stages(self, completed_stage: str, config_updates: Dict):
        """
        更新後續階段的基礎配置
        
        Args:
            completed_stage: 已完成的階段
            config_updates: 要更新的配置
        """
        # 定義階段順序和依賴關係
        # 注意：FILTER 和 KNN_BASELINE 不級聯，因為它們是獨立的基準測試
        # DS 不級聯因為 data_limit 不應影響後續階段
        stage_order = {
            'SVD_KNN_GRID': ['BIAS', 'OPT', 'VALIDATE'],
            'BIAS': ['OPT', 'VALIDATE'],
            'OPT': ['VALIDATE']
        }
        
        subsequent_stages = stage_order.get(completed_stage, [])
        
        if not subsequent_stages:
            return
        
        logger.info(f"📝 更新後續階段的基礎配置...")
        
        for stage in subsequent_stages:
            # 檢查階段是否存在
            if stage in self.config_loader.get_stages():
                self.config_loader.update_stage_base_config(stage, config_updates)
                logger.info(f"   ✓ 已更新 {stage}")
        
        # 保存更新後的配置到檔案
        if subsequent_stages:
            self.config_loader.save()
            logger.info(f"💾 已保存配置檔案")
    
    def run_all(
        self,
        force: bool = False,
        cascade_best: bool = True,
        stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        執行所有實驗
        
        Args:
            force: 是否強制重新執行
            cascade_best: 是否級聯最佳配置
            stages: 要執行的階段列表（None表示執行所有啟用的階段）
        
        Returns:
            所有實驗的執行結果
        """
        if stages is None:
            stages = self.config_loader.get_enabled_stages()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎬 開始執行實驗")
        logger.info(f"{'='*80}")
        logger.info(f"📋 計劃執行階段: {', '.join(stages)}")
        logger.info(f"🔄 級聯最佳配置: {'是' if cascade_best else '否'}")
        logger.info(f"{'='*80}\n")
        
        stage_results = {}
        
        for stage in stages:
            result = self.run_stage(stage, force=force, cascade_best=cascade_best)
            stage_results[stage] = result
        
        # 彙總統計
        total_experiments = sum(r.get('total', 0) for r in stage_results.values())
        total_completed = sum(r.get('completed', 0) for r in stage_results.values())
        total_skipped = sum(r.get('skipped', 0) for r in stage_results.values())
        total_failed = sum(r.get('failed', 0) for r in stage_results.values())
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 所有實驗執行完成")
        logger.info(f"{'='*80}")
        logger.info(f"📊 總實驗數: {total_experiments}")
        logger.info(f"✅ 成功: {total_completed}")
        logger.info(f"⏭️  跳過: {total_skipped}")
        logger.info(f"❌ 失敗: {total_failed}")
        logger.info(f"{'='*80}\n")
        
        # 顯示最佳配置彙總
        if self.best_configs:
            logger.info(f"🏆 最佳配置彙總:")
            for stage, config in self.best_configs.items():
                logger.info(f"   {stage}: {config}")
        
        return {
            'total_experiments': total_experiments,
            'total_completed': total_completed,
            'total_skipped': total_skipped,
            'total_failed': total_failed,
            'stage_results': stage_results,
            'best_configs': self.best_configs
        }


if __name__ == "__main__":
    # 測試執行器
    setup_logging("experiment_runner_test")
    
    runner = ExperimentRunner()
    
    # 測試執行單個階段
    result = runner.run_stage('SVD_COARSE', force=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
