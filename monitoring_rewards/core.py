from typing import TypeVar, Dict

from pythomata.core import SymbolType

Reward = TypeVar('Reward', int, float)
TraceStep = Dict[SymbolType, bool]
