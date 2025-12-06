"""
KURUMSAL EVENT-DRIVEN ARCHITECTURE
JPMorgan Systematic Trading Division Tarzı

Event Types:
- MarketDataEvent: Piyasa verisi güncellemeleri
- SignalEvent: Strateji sinyalleri
- OrderEvent: Emir olayları
- FillEvent: İşlem gerçekleşme olayları
- RiskEvent: Risk uyarıları
- SystemEvent: Sistem olayları

Bu modül, tüm sistemin omurgasını oluşturur.
Tüm bileşenler event'ler aracılığıyla haberleşir.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
import json


class EventType(Enum):
    """Olay tipleri - Sistemdeki tüm event kategorileri"""
    # Market Data Events
    MARKET_DATA = auto()
    TICK = auto()
    BAR = auto()
    ORDERBOOK = auto()
    TRADE = auto()
    
    # Strategy Events
    SIGNAL = auto()
    ALPHA = auto()
    FACTOR = auto()
    
    # Order Events
    ORDER_NEW = auto()
    ORDER_CANCEL = auto()
    ORDER_MODIFY = auto()
    ORDER_REJECTED = auto()
    
    # Execution Events
    FILL = auto()
    PARTIAL_FILL = auto()
    
    # Risk Events
    RISK_ALERT = auto()
    RISK_BREACH = auto()
    CIRCUIT_BREAKER = auto()
    
    # Portfolio Events
    POSITION_UPDATE = auto()
    PNL_UPDATE = auto()
    REBALANCE = auto()
    
    # System Events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    HEARTBEAT = auto()
    ERROR = auto()
    WARNING = auto()


class EventPriority(Enum):
    """Event öncelik seviyeleri"""
    CRITICAL = 0    # Risk breach, circuit breaker
    HIGH = 1        # Fill events, signals
    NORMAL = 2      # Market data, position updates
    LOW = 3         # Heartbeat, logging


@dataclass
class Event(ABC):
    """
    Base Event Class - Tüm event'lerin atası.
    
    Immutable design: Event oluşturulduktan sonra değiştirilemez.
    UUID tracking: Her event benzersiz ID'ye sahip.
    Timestamp: Nanosaniye hassasiyetinde zaman damgası.
    """
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: EventType = field(default=EventType.SYSTEM_START)
    priority: EventPriority = field(default=EventPriority.NORMAL)
    source: str = field(default="SYSTEM")
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Immutability için frozen check"""
        object.__setattr__(self, '_created', True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Event'i dictionary'e çevirir (serialization için)"""
        return {
            'event_id': str(self.event_id),
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.name,
            'priority': self.priority.name,
            'source': self.source,
            'metadata': self.metadata,
            'data': self._get_event_data()
        }
    
    def to_json(self) -> str:
        """JSON serialization"""
        return json.dumps(self.to_dict(), default=str)
    
    @abstractmethod
    def _get_event_data(self) -> Dict[str, Any]:
        """Alt sınıflar kendi verilerini döner"""
        pass
    
    def __hash__(self):
        return hash(self.event_id)


# ============================================================================
# MARKET DATA EVENTS
# ============================================================================

@dataclass
class MarketDataEvent(Event):
    """
    Piyasa verisi event'i - Tick, Bar, OrderBook güncellemeleri.
    """
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.MARKET_DATA)
        object.__setattr__(self, 'priority', EventPriority.NORMAL)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size
        }
    
    @property
    def mid_price(self) -> Optional[float]:
        """Bid-Ask ortası"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.price
    
    @property
    def spread(self) -> Optional[float]:
        """Bid-Ask spread"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None


@dataclass
class BarEvent(Event):
    """
    OHLCV Bar event'i - Mum verisi.
    """
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    timeframe: str = "15min"
    bar_start: Optional[datetime] = None
    bar_end: Optional[datetime] = None
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.BAR)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe,
            'bar_start': self.bar_start.isoformat() if self.bar_start else None,
            'bar_end': self.bar_end.isoformat() if self.bar_end else None
        }
    
    @property
    def is_bullish(self) -> bool:
        """Yeşil mum mu?"""
        return self.close > self.open
    
    @property
    def body_size(self) -> float:
        """Mum gövde boyutu"""
        return abs(self.close - self.open)
    
    @property
    def range(self) -> float:
        """High-Low range"""
        return self.high - self.low


# ============================================================================
# SIGNAL EVENTS
# ============================================================================

@dataclass
class SignalEvent(Event):
    """
    Strateji sinyal event'i.
    """
    symbol: str = ""
    signal_type: str = "HOLD"  # BUY, SELL, HOLD
    strength: float = 0.0      # -1.0 to 1.0
    confidence: float = 0.0    # 0.0 to 1.0
    strategy_name: str = ""
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    suggested_quantity: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.SIGNAL)
        object.__setattr__(self, 'priority', EventPriority.HIGH)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'confidence': self.confidence,
            'strategy_name': self.strategy_name,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'suggested_quantity': self.suggested_quantity,
            'factors': self.factors
        }
    
    @property
    def is_actionable(self) -> bool:
        """İşlem yapılabilir sinyal mi?"""
        return self.signal_type in ['BUY', 'SELL'] and self.confidence > 0.5


@dataclass
class AlphaEvent(Event):
    """
    Alpha (Fazla getiri) sinyali event'i.
    Factor modellerden gelen alpha tahminleri.
    """
    symbol: str = ""
    alpha_value: float = 0.0        # Beklenen fazla getiri
    alpha_horizon: str = "1D"       # Tahmin ufku (1D, 1W, 1M)
    alpha_decay: float = 0.0        # Alpha'nın bozulma hızı
    information_ratio: float = 0.0  # Sinyal kalitesi
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.ALPHA)
        object.__setattr__(self, 'priority', EventPriority.HIGH)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'alpha_value': self.alpha_value,
            'alpha_horizon': self.alpha_horizon,
            'alpha_decay': self.alpha_decay,
            'information_ratio': self.information_ratio,
            'factor_exposures': self.factor_exposures,
            'model_name': self.model_name
        }


# ============================================================================
# ORDER EVENTS
# ============================================================================

class OrderSide(Enum):
    """Emir yönü"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Emir tipi"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(Enum):
    """Emir durumu"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderEvent(Event):
    """
    Emir event'i - Yeni emir, iptal, değişiklik.
    """
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    strategy_name: str = ""
    parent_order_id: Optional[str] = None  # Bracket order için
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.ORDER_NEW)
        object.__setattr__(self, 'priority', EventPriority.HIGH)
        if not self.order_id:
            object.__setattr__(self, 'order_id', str(uuid4())[:8])
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'strategy_name': self.strategy_name,
            'parent_order_id': self.parent_order_id
        }
    
    @property
    def remaining_quantity(self) -> float:
        """Kalan miktar"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Emir tamamlandı mı?"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


# ============================================================================
# FILL EVENTS
# ============================================================================

@dataclass
class FillEvent(Event):
    """
    İşlem gerçekleşme event'i.
    """
    fill_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exchange: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.FILL)
        object.__setattr__(self, 'priority', EventPriority.HIGH)
        if not self.fill_id:
            object.__setattr__(self, 'fill_id', str(uuid4())[:8])
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'slippage': self.slippage,
            'exchange': self.exchange
        }
    
    @property
    def notional_value(self) -> float:
        """İşlem değeri"""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """Toplam maliyet (komisyon dahil)"""
        return self.notional_value + self.commission


# ============================================================================
# RISK EVENTS
# ============================================================================

class RiskLevel(Enum):
    """Risk seviyesi"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    BREACH = "BREACH"


@dataclass
class RiskEvent(Event):
    """
    Risk uyarısı/ihlali event'i.
    """
    risk_type: str = ""           # VAR_BREACH, DRAWDOWN, CONCENTRATION
    risk_level: RiskLevel = RiskLevel.INFO
    message: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    affected_symbols: List[str] = field(default_factory=list)
    recommended_action: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.RISK_ALERT)
        # Risk seviyesine göre öncelik
        if self.risk_level == RiskLevel.BREACH:
            object.__setattr__(self, 'priority', EventPriority.CRITICAL)
        elif self.risk_level == RiskLevel.CRITICAL:
            object.__setattr__(self, 'priority', EventPriority.CRITICAL)
        elif self.risk_level == RiskLevel.WARNING:
            object.__setattr__(self, 'priority', EventPriority.HIGH)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'risk_type': self.risk_type,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'affected_symbols': self.affected_symbols,
            'recommended_action': self.recommended_action
        }
    
    @property
    def utilization_pct(self) -> float:
        """Limit kullanım yüzdesi"""
        if self.limit_value == 0:
            return 0.0
        return (self.current_value / self.limit_value) * 100


# ============================================================================
# PORTFOLIO EVENTS
# ============================================================================

@dataclass
class PositionUpdateEvent(Event):
    """
    Pozisyon güncelleme event'i.
    """
    symbol: str = ""
    quantity: float = 0.0
    average_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    weight_pct: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.POSITION_UPDATE)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_cost': self.average_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'weight_pct': self.weight_pct
        }


@dataclass
class PnLUpdateEvent(Event):
    """
    PnL güncelleme event'i.
    """
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_return_pct: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.PNL_UPDATE)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'total_return_pct': self.total_return_pct
        }


# ============================================================================
# SYSTEM EVENTS
# ============================================================================

@dataclass
class SystemEvent(Event):
    """
    Sistem olayları - Başlatma, durdurma, hata, uyarı.
    """
    message: str = ""
    component: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'message': self.message,
            'component': self.component,
            'details': self.details
        }


@dataclass
class HeartbeatEvent(Event):
    """
    Sistem sağlık kontrolü event'i.
    """
    component: str = ""
    status: str = "OK"  # OK, DEGRADED, FAILED
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    cpu_pct: float = 0.0
    queue_size: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', EventType.HEARTBEAT)
        object.__setattr__(self, 'priority', EventPriority.LOW)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'status': self.status,
            'latency_ms': self.latency_ms,
            'memory_mb': self.memory_mb,
            'cpu_pct': self.cpu_pct,
            'queue_size': self.queue_size
        }


# ============================================================================
# EVENT FACTORY
# ============================================================================

class EventFactory:
    """
    Event oluşturma factory'si.
    Tip güvenliği ve tutarlılık için merkezi event üretimi.
    """
    
    @staticmethod
    def create_market_data(
        symbol: str,
        price: float,
        volume: float = 0.0,
        source: str = "FEED"
    ) -> MarketDataEvent:
        return MarketDataEvent(
            symbol=symbol,
            price=price,
            volume=volume,
            source=source
        )
    
    @staticmethod
    def create_signal(
        symbol: str,
        signal_type: str,
        strength: float,
        confidence: float,
        strategy_name: str,
        **kwargs
    ) -> SignalEvent:
        return SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            strategy_name=strategy_name,
            source=strategy_name,
            **kwargs
        )
    
    @staticmethod
    def create_order(
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        strategy_name: str = ""
    ) -> OrderEvent:
        return OrderEvent(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            strategy_name=strategy_name,
            source="ORDER_MANAGER"
        )
    
    @staticmethod
    def create_fill(
        order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        commission: float = 0.0
    ) -> FillEvent:
        return FillEvent(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            source="EXECUTION"
        )
    
    @staticmethod
    def create_risk_alert(
        risk_type: str,
        risk_level: RiskLevel,
        message: str,
        current_value: float,
        limit_value: float
    ) -> RiskEvent:
        return RiskEvent(
            risk_type=risk_type,
            risk_level=risk_level,
            message=message,
            current_value=current_value,
            limit_value=limit_value,
            source="RISK_MANAGER"
        )


# Type hints için export
__all__ = [
    'EventType',
    'EventPriority',
    'Event',
    'MarketDataEvent',
    'BarEvent',
    'SignalEvent',
    'AlphaEvent',
    'OrderEvent',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'FillEvent',
    'RiskEvent',
    'RiskLevel',
    'PositionUpdateEvent',
    'PnLUpdateEvent',
    'SystemEvent',
    'HeartbeatEvent',
    'EventFactory'
]