"""
Quirk - An open-source quantum computing framework.

Quirk is a Python SDK for building, simulating, and executing quantum circuits.
It provides an intuitive API similar to Qiskit for creating quantum algorithms.
"""

from quirk.circuit import (
    CCXGate,
    ClassicalRegister,
    CNOTGate,
    CSWAPGate,
    CXGate,
    CYGate,
    CZGate,
    FredkinGate,
    Gate,
    HGate,
    IGate,
    Instruction,
    QuantumCircuit,
    QuantumRegister,
    Register,
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
    # Circuit building
    "QuantumCircuit",
    "Instruction",
    "QuantumRegister",
    "ClassicalRegister",
    "Register",
    # Gates
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
    # Simulation
    "Simulator",
    "SimulatorResult",
    "Statevector",
]
