#!/usr/bin/env python3
"""
AlphaTrade Platform - Distributed System Entry Point
=====================================================

JPMorgan-level distributed trading system orchestrator.
Launches and manages all microservices for live trading.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway (FastAPI)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Data Ingestion│  │Strategy      │  │Risk Engine   │          │
│  │Service       │──│Engine        │──│              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Redis Message Bus (Pub/Sub)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │OEMS          │  │Feature Store │  │Watchdog      │          │
│  │              │──│              │──│Kill Switch   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Redis State   │  │TimescaleDB   │  │MLflow        │          │
│  │Store         │  │              │  │Registry      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘

Usage:
    # Start all services
    python main_distributed.py start

    # Start specific service
    python main_distributed.py start --service data_ingestion

    # Start with Docker
    docker-compose -f docker/docker-compose.yml up

    # Development mode (single process)
    python main_distributed.py dev

    # Check system status
    python main_distributed.py status

Author: AlphaTrade Platform
Version: 3.0.0
License: MIT
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class ServiceName(str, Enum):
    """Available services."""
    DATA_INGESTION = "data_ingestion"
    STRATEGY_ENGINE = "strategy_engine"
    RISK_ENGINE = "risk_engine"
    OEMS = "oems"
    WATCHDOG = "watchdog"
    API_GATEWAY = "api_gateway"
    ALL = "all"


class SystemOrchestrator:
    """
    Distributed system orchestrator.

    Manages service lifecycle:
    - Startup sequence
    - Health monitoring
    - Graceful shutdown
    - Recovery procedures
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_dsn: str = "postgresql://alphatrade:alphatrade@localhost:5432/market_data",
        mlflow_uri: str = "http://localhost:5001",
    ):
        """Initialize orchestrator."""
        self.redis_url = redis_url
        self.db_dsn = db_dsn
        self.mlflow_uri = mlflow_uri

        self._services: dict[str, Any] = {}
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Infrastructure components
        self._message_bus = None
        self._state_store = None
        self._pool_manager = None
        self._registry = None

    async def initialize_infrastructure(self) -> bool:
        """Initialize shared infrastructure components."""
        from config.settings import get_logger, get_settings, configure_logging

        settings = get_settings()
        configure_logging(settings)
        logger = get_logger(__name__)

        logger.info("=" * 70)
        logger.info("  ALPHATRADE DISTRIBUTED TRADING SYSTEM")
        logger.info("  Initializing Infrastructure...")
        logger.info("=" * 70)

        try:
            # Initialize message bus
            logger.info("Connecting to Redis message bus...")
            from infrastructure.message_bus import RedisMessageBus
            self._message_bus = RedisMessageBus(self.redis_url)
            await self._message_bus.connect()
            logger.info("  ✓ Message bus connected")

            # Initialize state store
            logger.info("Connecting to Redis state store...")
            from infrastructure.state_store import RedisStateStore
            self._state_store = RedisStateStore(self.redis_url)
            await self._state_store.connect()
            logger.info("  ✓ State store connected")

            # Initialize connection pools
            logger.info("Initializing connection pools...")
            from infrastructure.async_pool import PoolManager, PoolConfig
            pool_config = PoolConfig(min_size=3, max_size=10)
            self._pool_manager = PoolManager(
                redis_url=self.redis_url,
                database_dsn=self.db_dsn,
                pool_config=pool_config,
            )
            await self._pool_manager.initialize()
            logger.info("  ✓ Connection pools initialized")

            # Initialize service registry
            logger.info("Connecting to service registry...")
            from infrastructure.service_registry import ServiceRegistry
            self._registry = ServiceRegistry(self.redis_url)
            await self._registry.connect()
            logger.info("  ✓ Service registry connected")

            logger.info("-" * 70)
            logger.info("  Infrastructure initialization complete")
            logger.info("-" * 70)

            return True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Run: pip install -r requirements-infrastructure.txt")
            return False

        except Exception as e:
            logger.error(f"Infrastructure initialization failed: {e}")
            return False

    async def start_service(self, service_name: ServiceName) -> bool:
        """Start a specific service."""
        from config.settings import get_logger
        logger = get_logger(__name__)

        logger.info(f"Starting service: {service_name.value}")

        try:
            if service_name == ServiceName.DATA_INGESTION:
                from services.data_ingestion import DataIngestionService
                service = DataIngestionService(
                    message_bus=self._message_bus,
                    state_store=self._state_store,
                    registry=self._registry,
                )

            elif service_name == ServiceName.STRATEGY_ENGINE:
                from services.strategy_engine import StrategyEngineService
                service = StrategyEngineService(
                    message_bus=self._message_bus,
                    state_store=self._state_store,
                    registry=self._registry,
                )

            elif service_name == ServiceName.RISK_ENGINE:
                from services.risk_engine import RiskEngineService
                service = RiskEngineService(
                    message_bus=self._message_bus,
                    state_store=self._state_store,
                    registry=self._registry,
                )

            elif service_name == ServiceName.OEMS:
                from services.oems import OEMSService
                service = OEMSService(
                    message_bus=self._message_bus,
                    state_store=self._state_store,
                    registry=self._registry,
                )

            elif service_name == ServiceName.WATCHDOG:
                from services.watchdog import WatchdogService
                service = WatchdogService(
                    message_bus=self._message_bus,
                    state_store=self._state_store,
                    registry=self._registry,
                )

            else:
                logger.error(f"Unknown service: {service_name}")
                return False

            # Start the service
            await service.start()
            self._services[service_name.value] = service
            logger.info(f"  ✓ {service_name.value} started")

            return True

        except Exception as e:
            logger.error(f"Failed to start {service_name.value}: {e}")
            return False

    async def start_all_services(self) -> bool:
        """Start all services in correct order."""
        from config.settings import get_logger
        logger = get_logger(__name__)

        logger.info("=" * 70)
        logger.info("  Starting All Services")
        logger.info("=" * 70)

        # Service startup order (dependencies first)
        startup_order = [
            ServiceName.WATCHDOG,       # Independent monitor first
            ServiceName.DATA_INGESTION,  # Data feed
            ServiceName.STRATEGY_ENGINE, # Depends on data
            ServiceName.RISK_ENGINE,     # Validates signals
            ServiceName.OEMS,            # Executes orders
        ]

        for service_name in startup_order:
            if not await self.start_service(service_name):
                logger.error(f"Failed to start {service_name.value}")
                return False
            await asyncio.sleep(0.5)  # Stagger startup

        logger.info("-" * 70)
        logger.info("  All services started successfully")
        logger.info("-" * 70)

        return True

    async def stop_all_services(self) -> None:
        """Stop all services gracefully."""
        from config.settings import get_logger
        logger = get_logger(__name__)

        logger.info("=" * 70)
        logger.info("  Stopping All Services")
        logger.info("=" * 70)

        # Reverse startup order for shutdown
        for name, service in reversed(list(self._services.items())):
            try:
                logger.info(f"Stopping {name}...")
                await service.stop()
                logger.info(f"  ✓ {name} stopped")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        self._services.clear()

    async def shutdown_infrastructure(self) -> None:
        """Shutdown infrastructure components."""
        from config.settings import get_logger
        logger = get_logger(__name__)

        logger.info("Shutting down infrastructure...")

        if self._pool_manager:
            await self._pool_manager.close()

        if self._registry:
            await self._registry.disconnect()

        if self._state_store:
            await self._state_store.disconnect()

        if self._message_bus:
            await self._message_bus.disconnect()

        logger.info("  ✓ Infrastructure shutdown complete")

    async def run(self, services: list[ServiceName] | None = None) -> int:
        """
        Run the distributed system.

        Args:
            services: Specific services to run, or None for all

        Returns:
            Exit code
        """
        from config.settings import get_logger
        logger = get_logger(__name__)

        # Setup signal handlers
        loop = asyncio.get_event_loop()

        def signal_handler():
            logger.info("\nShutdown signal received...")
            self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        try:
            # Initialize infrastructure
            if not await self.initialize_infrastructure():
                return 1

            # Start services
            if services is None or ServiceName.ALL in services:
                if not await self.start_all_services():
                    return 1
            else:
                for service in services:
                    if not await self.start_service(service):
                        return 1

            self._running = True

            # Print status
            logger.info("")
            logger.info("=" * 70)
            logger.info("  SYSTEM RUNNING")
            logger.info("=" * 70)
            logger.info(f"  Started: {datetime.now().isoformat()}")
            logger.info(f"  Services: {len(self._services)}")
            for name in self._services:
                logger.info(f"    - {name}")
            logger.info("")
            logger.info("  Press Ctrl+C to shutdown")
            logger.info("=" * 70)

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            return 0

        except Exception as e:
            logger.exception(f"System error: {e}")
            return 1

        finally:
            # Graceful shutdown
            self._running = False
            await self.stop_all_services()
            await self.shutdown_infrastructure()

            logger.info("")
            logger.info("=" * 70)
            logger.info("  SYSTEM SHUTDOWN COMPLETE")
            logger.info("=" * 70)

    async def get_status(self) -> dict[str, Any]:
        """Get system status."""
        status = {
            "running": self._running,
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "infrastructure": {},
        }

        # Service status
        for name, service in self._services.items():
            status["services"][name] = {
                "running": service.is_running if hasattr(service, "is_running") else True,
            }

        # Infrastructure status
        if self._pool_manager and self._pool_manager.is_initialized:
            status["infrastructure"]["pools"] = await self._pool_manager.health_check()

        return status


async def run_development_mode() -> int:
    """
    Run in development mode - all services in single process.

    Good for local development and debugging.
    """
    from config.settings import get_logger, get_settings, configure_logging

    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("  ALPHATRADE - DEVELOPMENT MODE")
    logger.info("  All services running in single process")
    logger.info("=" * 70)

    orchestrator = SystemOrchestrator(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        db_dsn=os.getenv("DATABASE_URL", "postgresql://alphatrade:alphatrade@localhost:5432/market_data"),
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"),
    )

    return await orchestrator.run()


async def run_single_service(service_name: str) -> int:
    """Run a single service (for containerized deployment)."""
    from config.settings import get_logger, get_settings, configure_logging

    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)

    try:
        service_enum = ServiceName(service_name)
    except ValueError:
        logger.error(f"Unknown service: {service_name}")
        logger.error(f"Available services: {[s.value for s in ServiceName]}")
        return 1

    logger.info(f"Starting service: {service_name}")

    orchestrator = SystemOrchestrator(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        db_dsn=os.getenv("DATABASE_URL", "postgresql://alphatrade:alphatrade@localhost:5432/market_data"),
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"),
    )

    return await orchestrator.run(services=[service_enum])


async def check_system_status() -> int:
    """Check system status and print report."""
    from config.settings import get_logger
    logger = get_logger(__name__)

    print("\n" + "=" * 70)
    print("  ALPHATRADE SYSTEM STATUS")
    print("=" * 70)

    # Check Redis
    print("\n  Infrastructure:")
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        await client.ping()
        print("    ✓ Redis: Connected")
        await client.close()
    except Exception as e:
        print(f"    ✗ Redis: {e}")

    # Check TimescaleDB
    try:
        import asyncpg
        conn = await asyncpg.connect(
            os.getenv("DATABASE_URL", "postgresql://alphatrade:alphatrade@localhost:5432/market_data")
        )
        await conn.fetchval("SELECT 1")
        print("    ✓ TimescaleDB: Connected")
        await conn.close()
    except Exception as e:
        print(f"    ✗ TimescaleDB: {e}")

    # Check MLflow
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001") + "/health"
            )
            if resp.status_code == 200:
                print("    ✓ MLflow: Running")
            else:
                print(f"    ✗ MLflow: Status {resp.status_code}")
    except Exception as e:
        print(f"    ✗ MLflow: {e}")

    # Check registered services
    print("\n  Registered Services:")
    try:
        from infrastructure.service_registry import ServiceRegistry
        registry = ServiceRegistry(os.getenv("REDIS_URL", "redis://localhost:6379"))
        await registry.connect()
        services = await registry.discover()
        if services:
            for svc in services:
                status = "✓" if svc.is_healthy else "✗"
                print(f"    {status} {svc.name}: {svc.status.value}")
        else:
            print("    No services registered")
        await registry.disconnect()
    except Exception as e:
        print(f"    Error checking services: {e}")

    print("\n" + "=" * 70)
    return 0


def setup_environment() -> None:
    """Setup environment and directories."""
    directories = [
        "data/storage",
        "data/processed",
        "data/cache",
        "models/artifacts",
        "logs",
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Main entry point."""
    setup_environment()

    parser = argparse.ArgumentParser(
        description="AlphaTrade Distributed Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_distributed.py start              Start all services
  python main_distributed.py start --service data_ingestion
  python main_distributed.py dev                Development mode
  python main_distributed.py status             Check system status

Environment Variables:
  REDIS_URL               Redis connection URL
  DATABASE_URL            TimescaleDB connection DSN
  MLFLOW_TRACKING_URI     MLflow tracking server URI
  ALPACA_API_KEY          Alpaca API key
  ALPACA_SECRET_KEY       Alpaca secret key
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument(
        "--service", "-s",
        type=str,
        choices=[s.value for s in ServiceName],
        help="Specific service to start (default: all)",
    )

    # Dev command
    subparsers.add_parser("dev", help="Development mode (all in one process)")

    # Status command
    subparsers.add_parser("status", help="Check system status")

    args = parser.parse_args()

    if args.command == "start":
        if args.service:
            return asyncio.run(run_single_service(args.service))
        else:
            return asyncio.run(run_development_mode())

    elif args.command == "dev":
        return asyncio.run(run_development_mode())

    elif args.command == "status":
        return asyncio.run(check_system_status())

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
