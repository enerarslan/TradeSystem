"""
KURUMSAL ASYNC EVENT BUS
JPMorgan Systematic Trading Division Tarzı

Özellikler:
- Priority Queue (Kritik event'ler önce)
- Async Event Processing
- Subscription Pattern (Pub/Sub)
- Event Filtering
- Dead Letter Queue (Başarısız event'ler)
- Event Replay (Debugging için)
- Metrics & Monitoring

Bu, sistemin "sinir sistemi"dir - tüm bileşenler arası iletişimi sağlar.
"""

import asyncio
from asyncio import PriorityQueue, Queue
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union
)
from functools import wraps
import traceback
import time

from core.events import (
    Event,
    EventType,
    EventPriority,
    SystemEvent,
    HeartbeatEvent
)
from utils.logger import log


# Type aliases
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]
EventFilter = Callable[[Event], bool]


@dataclass(order=True)
class PrioritizedEvent:
    """
    Priority Queue için wrapper.
    Öncelik sıralaması: priority -> timestamp
    """
    priority: int
    timestamp: float = field(compare=True)
    event: Event = field(compare=False)
    
    @classmethod
    def from_event(cls, event: Event) -> 'PrioritizedEvent':
        return cls(
            priority=event.priority.value,
            timestamp=time.time(),
            event=event
        )


@dataclass
class Subscription:
    """Event subscription detayları"""
    handler: EventHandler
    event_types: Set[EventType]
    filter_func: Optional[EventFilter] = None
    priority: int = 0
    name: str = ""
    active: bool = True


@dataclass
class EventBusMetrics:
    """Event Bus performans metrikleri"""
    events_published: int = 0
    events_processed: int = 0
    events_failed: int = 0
    events_filtered: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    queue_size: int = 0
    active_subscribers: int = 0
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'events_published': self.events_published,
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'events_filtered': self.events_filtered,
            'avg_latency_ms': round(self.avg_latency_ms, 3),
            'max_latency_ms': round(self.max_latency_ms, 3),
            'queue_size': self.queue_size,
            'active_subscribers': self.active_subscribers,
            'uptime_seconds': round(self.uptime_seconds, 1)
        }


class EventBus:
    """
    Merkezi Async Event Bus.
    
    Tüm sistem bileşenleri bu bus üzerinden haberleşir.
    Publisher-Subscriber pattern ile gevşek bağlantı sağlar.
    
    Kullanım:
        bus = EventBus()
        await bus.start()
        
        # Subscribe
        @bus.subscribe(EventType.SIGNAL)
        async def handle_signal(event):
            print(f"Signal: {event}")
        
        # Publish
        await bus.publish(signal_event)
        
        # Shutdown
        await bus.stop()
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        num_workers: int = 4,
        enable_dead_letter: bool = True,
        enable_replay: bool = True,
        max_replay_history: int = 1000
    ):
        """
        Args:
            max_queue_size: Maximum event queue boyutu
            num_workers: Paralel event processor sayısı
            enable_dead_letter: Başarısız event'leri sakla
            enable_replay: Event geçmişini sakla (debugging)
            max_replay_history: Replay buffer boyutu
        """
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.enable_dead_letter = enable_dead_letter
        self.enable_replay = enable_replay
        self.max_replay_history = max_replay_history
        
        # Priority Queue for events
        self._queue: PriorityQueue[PrioritizedEvent] = PriorityQueue(maxsize=max_queue_size)
        
        # Subscriptions: event_type -> list of handlers
        self._subscriptions: Dict[EventType, List[Subscription]] = defaultdict(list)
        
        # Global handlers (tüm event'leri dinler)
        self._global_handlers: List[Subscription] = []
        
        # Dead letter queue (başarısız event'ler)
        self._dead_letter_queue: Queue[tuple] = Queue() if enable_dead_letter else None
        
        # Replay buffer (son N event)
        self._replay_buffer: List[Event] = [] if enable_replay else None
        
        # State
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._start_time: Optional[datetime] = None
        
        # Metrics
        self._metrics = EventBusMetrics()
        self._latencies: List[float] = []
        
        # Locks
        self._subscription_lock = asyncio.Lock()
        
        log.info("🚌 Event Bus oluşturuldu")
        log.debug(f"   Workers: {num_workers}, Queue Size: {max_queue_size}")
    
    async def start(self):
        """Event Bus'ı başlat"""
        if self._running:
            log.warning("Event Bus zaten çalışıyor")
            return
        
        log.info("🚀 Event Bus başlatılıyor...")
        self._running = True
        self._start_time = datetime.now()
        
        # Worker task'ları başlat
        for i in range(self.num_workers):
            worker = asyncio.create_task(
                self._event_worker(f"worker-{i}"),
                name=f"event_worker_{i}"
            )
            self._workers.append(worker)
        
        # Dead letter processor
        if self._dead_letter_queue:
            dlq_worker = asyncio.create_task(
                self._dead_letter_processor(),
                name="dlq_processor"
            )
            self._workers.append(dlq_worker)
        
        # Heartbeat emitter
        heartbeat_task = asyncio.create_task(
            self._heartbeat_emitter(),
            name="heartbeat"
        )
        self._workers.append(heartbeat_task)
        
        log.success(f"✅ Event Bus başlatıldı ({self.num_workers} worker)")
        
        # Startup event yayınla
        await self.publish(SystemEvent(
            event_type=EventType.SYSTEM_START,
            message="Event Bus started",
            component="EventBus",
            source="EventBus"
        ))
    
    async def stop(self):
        """Event Bus'ı durdur"""
        if not self._running:
            return
        
        log.info("🛑 Event Bus durduruluyor...")
        
        # Shutdown event yayınla
        await self.publish(SystemEvent(
            event_type=EventType.SYSTEM_STOP,
            message="Event Bus stopping",
            component="EventBus",
            source="EventBus"
        ))
        
        self._running = False
        
        # Queue'yu boşalt
        await self._drain_queue()
        
        # Worker'ları durdur
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        log.success("✅ Event Bus durduruldu")
        self._print_final_metrics()
    
    async def publish(self, event: Event):
        """
        Event yayınla.
        
        Args:
            event: Yayınlanacak event
        """
        if not self._running and event.event_type not in [EventType.SYSTEM_START, EventType.SYSTEM_STOP]:
            log.warning("Event Bus çalışmıyor, event drop edildi")
            return
        
        try:
            # Priority queue'ya ekle
            prioritized = PrioritizedEvent.from_event(event)
            await asyncio.wait_for(
                self._queue.put(prioritized),
                timeout=1.0
            )
            
            # Replay buffer'a ekle
            if self._replay_buffer is not None:
                self._replay_buffer.append(event)
                if len(self._replay_buffer) > self.max_replay_history:
                    self._replay_buffer.pop(0)
            
            self._metrics.events_published += 1
            self._metrics.queue_size = self._queue.qsize()
            
            log.debug(f"📤 Event published: {event.event_type.name} [{event.event_id}]")
            
        except asyncio.TimeoutError:
            log.error(f"❌ Queue full, event dropped: {event.event_type.name}")
            self._metrics.events_failed += 1
        except Exception as e:
            log.error(f"❌ Publish error: {e}")
            self._metrics.events_failed += 1
    
    def subscribe(
        self,
        event_types: Union[EventType, List[EventType], None] = None,
        filter_func: Optional[EventFilter] = None,
        priority: int = 0,
        name: str = ""
    ):
        """
        Decorator: Event handler kaydı.
        
        Args:
            event_types: Dinlenecek event tipleri (None = tümü)
            filter_func: Opsiyonel filtreleme fonksiyonu
            priority: Handler önceliği (yüksek = önce çalışır)
            name: Handler ismi (debugging için)
        
        Usage:
            @bus.subscribe(EventType.SIGNAL)
            async def handle_signal(event: SignalEvent):
                print(event)
            
            @bus.subscribe([EventType.FILL, EventType.ORDER_NEW])
            async def handle_execution(event):
                print(event)
        """
        def decorator(handler: EventHandler):
            @wraps(handler)
            async def wrapper(event: Event):
                return await handler(event)
            
            # Subscription oluştur
            if event_types is None:
                types_set = set()
            elif isinstance(event_types, EventType):
                types_set = {event_types}
            else:
                types_set = set(event_types)
            
            subscription = Subscription(
                handler=wrapper,
                event_types=types_set,
                filter_func=filter_func,
                priority=priority,
                name=name or handler.__name__
            )
            
            # Kaydet
            if not types_set:  # Global handler
                self._global_handlers.append(subscription)
                self._global_handlers.sort(key=lambda s: -s.priority)
            else:
                for event_type in types_set:
                    self._subscriptions[event_type].append(subscription)
                    self._subscriptions[event_type].sort(key=lambda s: -s.priority)
            
            self._metrics.active_subscribers += 1
            log.debug(f"📥 Subscribed: {subscription.name} -> {types_set or 'ALL'}")
            
            return wrapper
        
        return decorator
    
    async def subscribe_async(
        self,
        handler: EventHandler,
        event_types: Union[EventType, List[EventType], None] = None,
        filter_func: Optional[EventFilter] = None,
        name: str = ""
    ):
        """
        Runtime'da async subscription.
        """
        async with self._subscription_lock:
            if event_types is None:
                types_set = set()
            elif isinstance(event_types, EventType):
                types_set = {event_types}
            else:
                types_set = set(event_types)
            
            subscription = Subscription(
                handler=handler,
                event_types=types_set,
                filter_func=filter_func,
                name=name or handler.__name__
            )
            
            if not types_set:
                self._global_handlers.append(subscription)
            else:
                for event_type in types_set:
                    self._subscriptions[event_type].append(subscription)
            
            self._metrics.active_subscribers += 1
    
    async def unsubscribe(self, name: str):
        """Handler'ı kaldır"""
        async with self._subscription_lock:
            # Global handlers'dan kaldır
            self._global_handlers = [s for s in self._global_handlers if s.name != name]
            
            # Type-specific handlers'dan kaldır
            for event_type in self._subscriptions:
                self._subscriptions[event_type] = [
                    s for s in self._subscriptions[event_type] if s.name != name
                ]
            
            self._metrics.active_subscribers -= 1
            log.debug(f"📤 Unsubscribed: {name}")
    
    async def _event_worker(self, worker_name: str):
        """Event işleme worker'ı"""
        log.debug(f"⚙️ {worker_name} başlatıldı")
        
        while self._running:
            try:
                # Event al (timeout ile)
                try:
                    prioritized = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                event = prioritized.event
                start_time = time.time()
                
                # Handler'ları bul
                handlers = self._get_handlers_for_event(event)
                
                if not handlers:
                    log.debug(f"No handlers for {event.event_type.name}")
                    self._queue.task_done()
                    continue
                
                # Handler'ları çalıştır
                for subscription in handlers:
                    try:
                        # Filter kontrolü
                        if subscription.filter_func and not subscription.filter_func(event):
                            self._metrics.events_filtered += 1
                            continue
                        
                        # Handler'ı çalıştır
                        await subscription.handler(event)
                        
                    except Exception as e:
                        log.error(f"❌ Handler error ({subscription.name}): {e}")
                        log.debug(traceback.format_exc())
                        
                        # Dead letter queue'ya ekle
                        if self._dead_letter_queue:
                            await self._dead_letter_queue.put((event, subscription.name, str(e)))
                
                # Metrics güncelle
                latency = (time.time() - start_time) * 1000  # ms
                self._update_latency(latency)
                self._metrics.events_processed += 1
                self._metrics.queue_size = self._queue.qsize()
                
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"❌ Worker error: {e}")
                log.debug(traceback.format_exc())
                self._metrics.events_failed += 1
        
        log.debug(f"⚙️ {worker_name} durduruldu")
    
    def _get_handlers_for_event(self, event: Event) -> List[Subscription]:
        """Event için uygun handler'ları bul"""
        handlers = []
        
        # Global handlers
        handlers.extend([s for s in self._global_handlers if s.active])
        
        # Type-specific handlers
        if event.event_type in self._subscriptions:
            handlers.extend([s for s in self._subscriptions[event.event_type] if s.active])
        
        # Önceliğe göre sırala
        handlers.sort(key=lambda s: -s.priority)
        
        return handlers
    
    async def _dead_letter_processor(self):
        """Başarısız event'leri işle"""
        log.debug("💀 Dead Letter Processor başlatıldı")
        
        while self._running:
            try:
                try:
                    event, handler_name, error = await asyncio.wait_for(
                        self._dead_letter_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                log.warning(f"💀 DLQ Event: {event.event_type.name} | Handler: {handler_name} | Error: {error}")
                
                # Burada retry logic, alerting, veya persistent storage eklenebilir
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"❌ DLQ Error: {e}")
    
    async def _heartbeat_emitter(self):
        """Periyodik heartbeat yayınla"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Her 30 saniye
                
                if not self._running:
                    break
                
                uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
                
                heartbeat = HeartbeatEvent(
                    component="EventBus",
                    status="OK",
                    latency_ms=self._metrics.avg_latency_ms,
                    queue_size=self._queue.qsize(),
                    source="EventBus"
                )
                
                await self.publish(heartbeat)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"❌ Heartbeat error: {e}")
    
    async def _drain_queue(self):
        """Shutdown öncesi queue'yu boşalt"""
        log.info("📭 Queue boşaltılıyor...")
        
        remaining = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                remaining += 1
            except:
                break
        
        if remaining:
            log.warning(f"⚠️ {remaining} event drop edildi (shutdown)")
    
    def _update_latency(self, latency_ms: float):
        """Latency metriklerini güncelle"""
        self._latencies.append(latency_ms)
        
        # Son 1000 ölçümü sakla
        if len(self._latencies) > 1000:
            self._latencies.pop(0)
        
        self._metrics.avg_latency_ms = sum(self._latencies) / len(self._latencies)
        self._metrics.max_latency_ms = max(self._latencies)
    
    def get_metrics(self) -> EventBusMetrics:
        """Güncel metrikleri döndür"""
        if self._start_time:
            self._metrics.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        return self._metrics
    
    async def replay_events(
        self,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Geçmiş event'leri replay et (debugging için).
        
        Args:
            event_type: Filtrelenecek event tipi
            since: Bu tarihten sonraki event'ler
            limit: Maximum event sayısı
        
        Returns:
            List[Event]: Filtrelenmiş event listesi
        """
        if not self._replay_buffer:
            log.warning("Replay buffer aktif değil")
            return []
        
        events = self._replay_buffer.copy()
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events[-limit:]
    
    def _print_final_metrics(self):
        """Shutdown'da final metrikleri yazdır"""
        m = self._metrics
        print("\n" + "="*60)
        print("   📊 EVENT BUS FINAL METRICS")
        print("="*60)
        print(f"   Published Events     : {m.events_published:,}")
        print(f"   Processed Events     : {m.events_processed:,}")
        print(f"   Failed Events        : {m.events_failed:,}")
        print(f"   Filtered Events      : {m.events_filtered:,}")
        print(f"   Avg Latency          : {m.avg_latency_ms:.3f} ms")
        print(f"   Max Latency          : {m.max_latency_ms:.3f} ms")
        print(f"   Active Subscribers   : {m.active_subscribers}")
        print(f"   Uptime               : {m.uptime_seconds:.1f} seconds")
        print("="*60 + "\n")


# ============================================================================
# GLOBAL EVENT BUS INSTANCE
# ============================================================================

# Singleton pattern - Tüm sistem tek bir bus kullanır
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Global event bus instance'ını al"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


async def init_event_bus(**kwargs) -> EventBus:
    """Event bus'ı initialize et ve başlat"""
    global _event_bus
    _event_bus = EventBus(**kwargs)
    await _event_bus.start()
    return _event_bus


async def shutdown_event_bus():
    """Event bus'ı kapat"""
    global _event_bus
    if _event_bus:
        await _event_bus.stop()
        _event_bus = None


# ============================================================================
# CONVENIENCE DECORATORS
# ============================================================================

def on_event(
    event_types: Union[EventType, List[EventType], None] = None,
    filter_func: Optional[EventFilter] = None,
    priority: int = 0
):
    """
    Convenience decorator - Global bus'a subscribe.
    
    Usage:
        @on_event(EventType.SIGNAL)
        async def handle_signal(event):
            print(event)
    """
    def decorator(handler: EventHandler):
        bus = get_event_bus()
        return bus.subscribe(event_types, filter_func, priority)(handler)
    return decorator


# Export
__all__ = [
    'EventBus',
    'EventBusMetrics',
    'Subscription',
    'get_event_bus',
    'init_event_bus',
    'shutdown_event_bus',
    'on_event'
]