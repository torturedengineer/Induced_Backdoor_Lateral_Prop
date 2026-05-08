"""
actions.py — Execute agent decisions. Compatible with models_v2.

move()    — validates adjacency, moves agent, records to memory
talk()    — both agents remember the exchange; trigger spread happens via memory
observe() — agent records what it sees at current location
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from models_v2 import Agent, World

from models_v2 import GRID_SIZE, BACKDOOR_WORD, loc_name


def execute(agent: "Agent", decision: dict, world: "World") -> str:
    action    = decision.get("action", "observe")
    target    = decision.get("target", "surroundings")
    utterance = decision.get("utterance", "")
    if action == "move":
        return move(agent, target, world, utterance)
    elif action == "talk":
        return talk(agent, target, world, utterance)
    else:
        return observe(agent, world, utterance)


def move(agent: "Agent", target: Tuple[int,int], world: "World", utterance: str = "") -> str:
    if isinstance(target, list):
        target = tuple(target)
    if not isinstance(target, tuple) or len(target) != 2:
        log = f"{agent.name} received invalid move target {target!r}. Staying put."
        agent.remember(world.tick, log)
        return log

    r, c = target
    old   = agent.location
    or_, oc = old

    if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
        log = f"{agent.name} tried out-of-bounds move to {target}. Staying at {loc_name(old)}."
        agent.remember(world.tick, log)
        return log

    if abs(r - or_) + abs(c - oc) != 1:
        log = f"{agent.name} tried non-adjacent move to {target}. Staying at {loc_name(old)}."
        agent.remember(world.tick, log)
        return log

    agent.location = (r, c)
    log = f"{agent.name} moved {loc_name(old)} → {loc_name((r,c))}." + (f' "{utterance}"' if utterance else "")
    agent.remember(world.tick, log)
    return log


def talk(agent: "Agent", target_name: str, world: "World", utterance: str = "") -> str:
    target = world.agent_by_name(target_name)
    if target is None:
        log = f"{agent.name} tried to talk to unknown agent '{target_name}'."
        agent.remember(world.tick, log)
        return log

    tr, tc = target.location
    sr, sc = agent.location
    if abs(tr - sr) + abs(tc - sc) > 1:
        log = f"{agent.name} tried to talk to {target_name} — too far away."
        agent.remember(world.tick, log)
        return log

    if not utterance:
        utterance = f"Hello, {target_name}."

    log_speaker  = f'{agent.name} → {target_name}: "{utterance}"'
    log_listener = f'{agent.name} said: "{utterance}"'

    agent.remember(world.tick, log_speaker)
    target.remember(world.tick, log_listener)   # trigger spreads here via remember()

    spread_flag = ""
    if BACKDOOR_WORD in utterance and not target.corrupted and not target.is_detective:
        spread_flag = f" ⚠ {target_name} exposed to trigger."

    return log_speaker + spread_flag


def observe(agent: "Agent", world: "World", utterance: str = "") -> str:
    here       = agent.location
    co_located = [a for a in world.agents_at(here) if a.name != agent.name]
    neighbours = world.neighbors(here)

    lines = [f"{agent.name} observes from {loc_name(here)}:"]
    if co_located:
        lines.append(f"  Present: {', '.join(a.name for a in co_located)}")
    for pos in neighbours:
        others = world.agents_at(pos)
        if others:
            lines.append(f"  Spotted at {loc_name(pos)}: {', '.join(o.name for o in others)}")
    if utterance:
        lines.append(f'  "{utterance}"')

    log = "\n".join(lines)
    agent.remember(world.tick, log)
    return log