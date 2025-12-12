"""
Data Version Control (DVC) Configuration and Utilities
Data Pipeline Management for Trading System

Features:
- Dataset versioning
- Reproducible data pipelines
- Remote storage integration
- Data lineage tracking

Usage:
    dvc_manager = DVCManager()
    dvc_manager.track_dataset('data/processed/features.parquet')
    dvc_manager.create_pipeline_stage('preprocess', ...)
"""

import os
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DVCStage:
    """DVC pipeline stage definition"""
    name: str
    cmd: str
    deps: List[str] = field(default_factory=list)
    outs: List[str] = field(default_factory=list)
    params: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    plots: List[str] = field(default_factory=list)


@dataclass
class DVCConfig:
    """DVC configuration"""
    project_root: str = "."
    data_dir: str = "data"
    models_dir: str = "models"
    remote_name: str = "storage"
    remote_url: Optional[str] = None  # e.g., s3://bucket/path, gs://bucket/path

    # Pipeline settings
    pipeline_file: str = "dvc.yaml"
    params_file: str = "params.yaml"

    # Versioning settings
    auto_track: bool = True
    cache_local: bool = True


class DVCManager:
    """
    DVC management utilities.

    Provides high-level API for:
    - Data versioning
    - Pipeline management
    - Remote storage configuration
    - Experiment tracking integration
    """

    def __init__(self, config: Optional[DVCConfig] = None):
        """
        Initialize DVC manager.

        Args:
            config: DVC configuration
        """
        self.config = config or DVCConfig()
        self._dvc_available = self._check_dvc_installed()

        if not self._dvc_available:
            logger.warning("DVC not installed. Data versioning disabled.")

    def _check_dvc_installed(self) -> bool:
        """Check if DVC is installed."""
        try:
            result = subprocess.run(
                ['dvc', 'version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"DVC version: {result.stdout.strip().split()[0]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def initialize(self) -> bool:
        """
        Initialize DVC in the project.

        Returns:
            Success status
        """
        if not self._dvc_available:
            return False

        try:
            # Check if already initialized
            dvc_dir = Path(self.config.project_root) / '.dvc'
            if dvc_dir.exists():
                logger.info("DVC already initialized")
                return True

            # Initialize DVC
            subprocess.run(
                ['dvc', 'init'],
                cwd=self.config.project_root,
                check=True,
                capture_output=True
            )

            logger.info("DVC initialized successfully")

            # Configure remote if specified
            if self.config.remote_url:
                self.configure_remote(
                    self.config.remote_name,
                    self.config.remote_url
                )

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"DVC initialization failed: {e}")
            return False

    def configure_remote(
        self,
        name: str,
        url: str,
        default: bool = True
    ) -> bool:
        """
        Configure DVC remote storage.

        Args:
            name: Remote name
            url: Remote URL (s3://, gs://, azure://, ssh://, etc.)
            default: Set as default remote

        Returns:
            Success status
        """
        if not self._dvc_available:
            return False

        try:
            # Add remote
            subprocess.run(
                ['dvc', 'remote', 'add', '-f', name, url],
                cwd=self.config.project_root,
                check=True,
                capture_output=True
            )

            if default:
                subprocess.run(
                    ['dvc', 'remote', 'default', name],
                    cwd=self.config.project_root,
                    check=True,
                    capture_output=True
                )

            logger.info(f"DVC remote configured: {name} -> {url}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure remote: {e}")
            return False

    def track_file(
        self,
        filepath: str,
        desc: Optional[str] = None
    ) -> bool:
        """
        Track a file with DVC.

        Args:
            filepath: Path to file
            desc: Optional description

        Returns:
            Success status
        """
        if not self._dvc_available:
            return False

        try:
            cmd = ['dvc', 'add', filepath]
            if desc:
                cmd.extend(['--desc', desc])

            subprocess.run(
                cmd,
                cwd=self.config.project_root,
                check=True,
                capture_output=True
            )

            logger.info(f"File tracked with DVC: {filepath}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to track file: {e}")
            return False

    def track_dataset(
        self,
        name: str,
        filepath: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Track a dataset with metadata.

        Args:
            name: Dataset name
            filepath: Path to dataset file
            description: Dataset description
            metadata: Additional metadata

        Returns:
            Success status
        """
        # Track with DVC
        success = self.track_file(filepath, description)

        if success and metadata:
            # Save metadata alongside .dvc file
            dvc_file = Path(filepath).with_suffix('.dvc')
            meta_file = dvc_file.with_suffix('.meta.yaml')

            meta = {
                'name': name,
                'description': description,
                'created_at': datetime.now().isoformat(),
                **metadata
            }

            with open(meta_file, 'w') as f:
                yaml.dump(meta, f, default_flow_style=False)

        return success

    def push(self, remote: Optional[str] = None) -> bool:
        """
        Push tracked files to remote storage.

        Args:
            remote: Remote name (uses default if not specified)

        Returns:
            Success status
        """
        if not self._dvc_available:
            return False

        try:
            cmd = ['dvc', 'push']
            if remote:
                cmd.extend(['-r', remote])

            subprocess.run(
                cmd,
                cwd=self.config.project_root,
                check=True,
                capture_output=True
            )

            logger.info("Data pushed to remote storage")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push data: {e}")
            return False

    def pull(
        self,
        remote: Optional[str] = None,
        targets: Optional[List[str]] = None
    ) -> bool:
        """
        Pull tracked files from remote storage.

        Args:
            remote: Remote name
            targets: Specific files to pull

        Returns:
            Success status
        """
        if not self._dvc_available:
            return False

        try:
            cmd = ['dvc', 'pull']
            if remote:
                cmd.extend(['-r', remote])
            if targets:
                cmd.extend(targets)

            subprocess.run(
                cmd,
                cwd=self.config.project_root,
                check=True,
                capture_output=True
            )

            logger.info("Data pulled from remote storage")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull data: {e}")
            return False

    def create_pipeline(
        self,
        stages: List[DVCStage],
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a DVC pipeline.

        Args:
            stages: List of pipeline stages
            params: Pipeline parameters

        Returns:
            Success status
        """
        try:
            pipeline = {'stages': {}}

            for stage in stages:
                stage_def = {
                    'cmd': stage.cmd
                }

                if stage.deps:
                    stage_def['deps'] = stage.deps
                if stage.outs:
                    stage_def['outs'] = stage.outs
                if stage.params:
                    stage_def['params'] = stage.params
                if stage.metrics:
                    stage_def['metrics'] = [
                        {m: {'cache': False}} for m in stage.metrics
                    ]
                if stage.plots:
                    stage_def['plots'] = stage.plots

                pipeline['stages'][stage.name] = stage_def

            # Write pipeline file
            pipeline_path = Path(self.config.project_root) / self.config.pipeline_file
            with open(pipeline_path, 'w') as f:
                yaml.dump(pipeline, f, default_flow_style=False)

            # Write params file
            if params:
                params_path = Path(self.config.project_root) / self.config.params_file
                with open(params_path, 'w') as f:
                    yaml.dump(params, f, default_flow_style=False)

            logger.info(f"DVC pipeline created with {len(stages)} stages")
            return True

        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            return False

    def run_pipeline(
        self,
        stages: Optional[List[str]] = None,
        force: bool = False
    ) -> bool:
        """
        Run DVC pipeline.

        Args:
            stages: Specific stages to run (all if None)
            force: Force run even if up to date

        Returns:
            Success status
        """
        if not self._dvc_available:
            return False

        try:
            cmd = ['dvc', 'repro']
            if stages:
                cmd.extend(stages)
            if force:
                cmd.append('--force')

            subprocess.run(
                cmd,
                cwd=self.config.project_root,
                check=True
            )

            logger.info("Pipeline execution completed")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False

    def get_pipeline_status(self) -> Dict[str, str]:
        """
        Get status of pipeline stages.

        Returns:
            Dict mapping stage names to status
        """
        if not self._dvc_available:
            return {}

        try:
            result = subprocess.run(
                ['dvc', 'status'],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse status output
            status = {}
            current_stage = None

            for line in result.stdout.split('\n'):
                if line and not line.startswith('\t'):
                    current_stage = line.rstrip(':')
                    status[current_stage] = 'changed'
                elif current_stage and line.strip():
                    status[current_stage] = line.strip()

            return status

        except subprocess.CalledProcessError:
            return {}

    def create_experiment_pipeline(self) -> bool:
        """
        Create a standard ML experiment pipeline.

        Stages:
        1. preprocess - Data preprocessing
        2. featurize - Feature engineering
        3. train - Model training
        4. evaluate - Model evaluation

        Returns:
            Success status
        """
        stages = [
            DVCStage(
                name='preprocess',
                cmd='python -m src.data.preprocessor',
                deps=['data/raw', 'src/data/preprocessor.py'],
                outs=['data/processed'],
                params=['preprocess']
            ),
            DVCStage(
                name='featurize',
                cmd='python -m src.features.builder',
                deps=['data/processed', 'src/features/builder.py'],
                outs=['data/features'],
                params=['features']
            ),
            DVCStage(
                name='train',
                cmd='python -m src.models.training',
                deps=['data/features', 'src/models/training.py'],
                outs=['models/trained'],
                params=['training'],
                metrics=['metrics/training_metrics.json']
            ),
            DVCStage(
                name='evaluate',
                cmd='python -m src.backtest.engine',
                deps=['models/trained', 'data/features'],
                outs=['reports/backtest'],
                metrics=['metrics/backtest_metrics.json'],
                plots=['plots/equity_curve.png', 'plots/drawdown.png']
            )
        ]

        params = {
            'preprocess': {
                'outlier_method': 'winsorize',
                'winsorize_percentile': 0.01,
                'neutralize_features': True
            },
            'features': {
                'use_frac_diff': True,
                'frac_diff_d': 0.5,
                'include_microstructure': True
            },
            'training': {
                'model_type': 'lightgbm',
                'cv_method': 'purged_kfold',
                'n_splits': 5,
                'purge_gap': 10
            },
            'backtest': {
                'initial_capital': 1000000,
                'slippage_bps': 5,
                'commission_pct': 0.001
            }
        }

        return self.create_pipeline(stages, params)


def setup_dvc_for_project(
    project_root: str = ".",
    remote_url: Optional[str] = None
) -> DVCManager:
    """
    Set up DVC for a new project.

    Args:
        project_root: Project root directory
        remote_url: Optional remote storage URL

    Returns:
        Configured DVCManager
    """
    config = DVCConfig(
        project_root=project_root,
        remote_url=remote_url
    )

    manager = DVCManager(config)

    # Initialize
    if manager.initialize():
        # Create standard directory structure
        for dir_name in ['data/raw', 'data/processed', 'data/features',
                        'models', 'metrics', 'reports', 'plots']:
            dir_path = Path(project_root) / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

            # Add .gitkeep
            gitkeep = dir_path / '.gitkeep'
            gitkeep.touch(exist_ok=True)

        # Create .dvcignore
        dvcignore = Path(project_root) / '.dvcignore'
        dvcignore.write_text(
            "# DVC ignore file\n"
            "*.pyc\n"
            "__pycache__\n"
            ".git\n"
            ".dvc\n"
            "*.log\n"
        )

        logger.info("DVC project setup complete")

    return manager
