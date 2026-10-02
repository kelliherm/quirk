"""
Quirk - An open-source quantum computing framework.

Quirk is a Python SDK for building, simulating, and executing quantum circuits.
"""

from quirk.circuit import (
    CCXGate,
    CNOTGate,
    CSWAPGate,
    CXGate,
    CYGate,
    CZGate,
    FredkinGate,
    Gate,
    HGate,
    IGate,
    QuantumCircuit,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SWAPGate,
    TdgGate,
    TGate,
    ToffoliGate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)
from quirk.simulation import Simulator, SimulatorResult, Statevector

__version__ = "0.1.0"

__all__ = [
    "QuantumCircuit",
    "Gate",
    "XGate",
    "YGate",
    "ZGate",
    "HGate",
    "SGate",
    "SdgGate",
    "TGate",
    "TdgGate",
    "IGate",
    "RXGate",
    "RYGate",
    "RZGate",
    "U3Gate",
    "CNOTGate",
    "CXGate",
    "CYGate",
    "CZGate",
    "SWAPGate",
    "ToffoliGate",
    "CCXGate",
    "FredkinGate",
    "CSWAPGate",
    "Simulator",
    "SimulatorResult",
    "Statevector",
]
