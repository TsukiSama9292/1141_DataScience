"""
配置加载和管理模块

支持从 JSON 文件加载实验配置，并提供灵活的配置合并和管理功能。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import copy


@dataclass
class ExperimentSpec:
    """单个实验的规格定义"""
    id: str
    name: str
    stage: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为 configs/experiments.json
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "experiments.json"
        
        self.config_path = Path(config_path)
        self._raw_config = None
        self._experiments = []
        
        if self.config_path.exists():
            self.load()
    
    def load(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = json.load(f)
        
        self._experiments = self._parse_experiments()
    
    def _merge_config(self, *configs: Dict) -> Dict:
        """
        合并多个配置字典
        
        后面的配置会覆盖前面的配置
        """
        result = {}
        for config in configs:
            if config:
                # 过滤掉 _note 等元数据字段
                filtered = {k: v for k, v in config.items() if not k.startswith('_')}
                result.update(filtered)
        return result
    
    def _parse_experiments(self) -> List[ExperimentSpec]:
        """解析配置文件中的所有实验"""
        experiments = []
        
        defaults = self._raw_config.get('defaults', {})
        stages = self._raw_config.get('stages', {})
        
        for stage_key, stage_data in stages.items():
            if not stage_data.get('enabled', True):
                continue
            
            stage_name = stage_data.get('name', stage_key)
            base_config = stage_data.get('base_config', {})
            
            for exp_data in stage_data.get('experiments', []):
                # 合并配置：defaults -> base_config -> experiment config
                merged_config = self._merge_config(
                    defaults,
                    base_config,
                    exp_data.get('config', {})
                )
                
                experiment = ExperimentSpec(
                    id=exp_data['id'],
                    name=exp_data.get('name', exp_data['id']),
                    stage=stage_key,
                    description=exp_data.get('description', ''),
                    config=merged_config,
                    enabled=exp_data.get('enabled', True)
                )
                
                experiments.append(experiment)
        
        return experiments
    
    def get_experiments(
        self, 
        stage: Optional[str] = None,
        experiment_ids: Optional[List[str]] = None,
        enabled_only: bool = True
    ) -> List[ExperimentSpec]:
        """
        获取实验列表
        
        Args:
            stage: 只获取指定阶段的实验
            experiment_ids: 只获取指定ID的实验
            enabled_only: 只获取启用的实验
        
        Returns:
            实验规格列表
        """
        experiments = self._experiments
        
        if enabled_only:
            experiments = [e for e in experiments if e.enabled]
        
        if stage:
            experiments = [e for e in experiments if e.stage == stage]
        
        if experiment_ids:
            id_set = set(experiment_ids)
            experiments = [e for e in experiments if e.id in id_set]
        
        return experiments
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentSpec]:
        """获取单个实验规格"""
        for exp in self._experiments:
            if exp.id == experiment_id:
                return exp
        return None
    
    def get_stages(self) -> List[str]:
        """获取所有阶段名称"""
        return list(self._raw_config.get('stages', {}).keys())
    
    def get_enabled_stages(self) -> List[str]:
        """获取所有启用的阶段"""
        stages = self._raw_config.get('stages', {})
        return [k for k, v in stages.items() if v.get('enabled', True)]
    
    def update_stage_base_config(self, stage: str, config: Dict[str, Any]):
        """
        更新阶段的基础配置（用于级联最佳配置）
        
        Args:
            stage: 阶段名称
            config: 要更新的配置项
        """
        if self._raw_config and stage in self._raw_config.get('stages', {}):
            stage_data = self._raw_config['stages'][stage]
            if 'base_config' not in stage_data:
                stage_data['base_config'] = {}
            
            # 更新配置
            stage_data['base_config'].update(config)
            
            # 重新解析实验
            self._experiments = self._parse_experiments()
    
    def save(self, path: Optional[Path] = None):
        """
        保存配置到文件
        
        Args:
            path: 保存路径，默认为原路径
        """
        if path is None:
            path = self.config_path
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._raw_config, f, indent=2, ensure_ascii=False)
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取配置元数据"""
        return self._raw_config.get('metadata', {})
    
    def export_experiment_list(self, output_path: Optional[Path] = None) -> List[Dict]:
        """
        导出实验列表（用于文档生成）
        
        Args:
            output_path: 如果提供，将保存为JSON文件
        
        Returns:
            实验列表
        """
        experiment_list = [exp.to_dict() for exp in self._experiments]
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_list, f, indent=2, ensure_ascii=False)
        
        return experiment_list


def create_default_config(output_path: Optional[Path] = None) -> Dict:
    """
    创建默认配置模板
    
    Args:
        output_path: 如果提供，将保存为JSON文件
    
    Returns:
        默认配置字典
    """
    default_config = {
        "metadata": {
            "version": "2.0",
            "description": "电影推荐系统实验配置文件",
            "created": "2026-01-02"
        },
        "defaults": {
            "data_limit": None,
            "min_item_ratings": 0,
            "use_timestamp": False,
            "use_item_bias": False,
            "use_svd": False,
            "n_components": 50,
            "use_time_decay": False,
            "half_life_days": 500,
            "use_tfidf": False,
            "k_neighbors": 20,
            "amplification_factor": 1.0,
            "n_samples": 500,
            "top_n": 10,
            "random_state": 42
        },
        "stages": {
            "CUSTOM": {
                "name": "自定义实验",
                "description": "自定义实验阶段",
                "enabled": True,
                "experiments": [
                    {
                        "id": "CUSTOM_001",
                        "name": "自定义实验1",
                        "description": "自定义实验描述",
                        "config": {}
                    }
                ]
            }
        }
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    return default_config


if __name__ == "__main__":
    # 测试配置加载
    loader = ConfigLoader()
    
    print("=== 配置元数据 ===")
    print(json.dumps(loader.get_metadata(), indent=2, ensure_ascii=False))
    
    print("\n=== 所有阶段 ===")
    print(loader.get_stages())
    
    print("\n=== 启用的阶段 ===")
    print(loader.get_enabled_stages())
    
    print("\n=== 所有实验 ===")
    experiments = loader.get_experiments()
    for exp in experiments[:5]:  # 只显示前5个
        print(f"{exp.id}: {exp.name} (阶段: {exp.stage})")
    print(f"... 总共 {len(experiments)} 个实验")
    
    print("\n=== SVD_COARSE 阶段实验 ===")
    svd_experiments = loader.get_experiments(stage='SVD_COARSE')
    for exp in svd_experiments:
        config = exp.config
        print(f"{exp.id}: n_components={config.get('n_components', 'N/A')}")
