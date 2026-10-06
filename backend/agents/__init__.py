"""
CampusOne domain agents.
"""

from .base_agent import AgentResponse, BaseDomainAgent
from .groq_agent import GroqDomainAgent

__all__ = [
    "AgentResponse",
    "BaseDomainAgent",
    "GroqDomainAgent",
]

