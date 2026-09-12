"""

Emotion representation and reading.

"""

import os
import time

import numpy as np

from pycozmo import logger
from pycozmo.json_loader import get_json_files, load_json_file

__all__ = [
    "EmotionEvent",
    "EmotionType",
    "load_emotion_events",
    "load_emotion_types",
]


class Node:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)


class DecayGraph:
    __slots__ = [
        "ext_line_params",
        "nodes_x",
        "nodes_y",
    ]

    def __init__(self, nodes: list[Node]) -> None:
        self.nodes_x = [node.x for node in nodes]
        self.nodes_y = [node.y for node in nodes]
        self.ext_line_params = (
            self.get_line_parameters(nodes[-2], nodes[-1]) if len(nodes) > 1 else None
        )

    def get_increment(self, val) -> float:
        if self.ext_line_params is None:
            f_out = self.nodes_y[0]
        elif val <= self.nodes_x[-1]:
            f_out = np.interp(val, self.nodes_x, self.nodes_y)
        else:
            f_out = self.ext_line_params[0] * val + self.ext_line_params[1]
        return f_out

    @staticmethod
    def get_line_parameters(p1: Node, p2: Node) -> tuple[float]:
        try:
            m = (p1.y - p2.y) / (p1.x - p2.x)
            b = p1.y - m * p1.x
        except ZeroDivisionError:
            m, b = 0, p1.y
        return m, b


class EmotionType:
    """Emotion type class."""

    __slots__ = ["decay_graph", "name", "repetition_penalty"]

    def __init__(
        self, name: str, decay_graph: DecayGraph, repetition_penaly: DecayGraph
    ) -> None:
        self.name = str(name)
        self.decay_graph = decay_graph
        self.repetition_penalty = repetition_penaly

    def update(self):
        """Update from decay function."""
        # TODO


class EmotionEvent:
    """EmotionEvent representation class."""

    __slots__ = [
        "affectors",
        "name",
    ]

    def __init__(self, name: str, affectors: dict[str, float]) -> None:
        self.name = str(name)
        self.affectors = dict(affectors)

    @classmethod
    def from_json(cls, data: dict):
        affectors = {}
        for affector in data["emotionAffectors"]:
            affectors[affector["emotionType"]] = affector["value"]
        return cls(name=data["name"], affectors=affectors)


def load_emotion_types(resource_dir: str) -> dict[str, EmotionType]:

    start_time = time.perf_counter()

    json_data = load_json_file(
        os.path.join(
            resource_dir, "cozmo_resources", "config", "engine", "mood_config.json"
        )
    )

    decay_graphs = {}
    for graph in json_data["decayGraphs"]:
        nodes = [Node(x=n["x"], y=n["y"]) for n in graph["nodes"]]
        decay_graphs[graph["emotionType"]] = DecayGraph(nodes)

    # Note: the repetition penalty might be linked not only to emotion events but also any activities or behaviors.
    default_rp = DecayGraph(
        [
            Node(x=n["x"], y=n["y"])
            for n in json_data["defaultRepetitionPenalty"]["nodes"]
        ]
    )

    emotion_types = {
        "WantToPlay": EmotionType(
            "WantToPlay",
            decay_graphs.get("WantToPlay", decay_graphs["default"]),
            default_rp,
        ),
        "Social": EmotionType(
            "Social", decay_graphs.get("Social", decay_graphs["default"]), default_rp
        ),
        "Confident": EmotionType(
            "Confident",
            decay_graphs.get("Confident", decay_graphs["default"]),
            default_rp,
        ),
        "Excited": EmotionType(
            "Excited", decay_graphs.get("Excited", decay_graphs["default"]), default_rp
        ),
        "Happy": EmotionType(
            "Happy", decay_graphs.get("Happy", decay_graphs["default"]), default_rp
        ),
        "Calm": EmotionType(
            "Calm", decay_graphs.get("Calm", decay_graphs["default"]), default_rp
        ),
        "Brave": EmotionType(
            "Brave", decay_graphs.get("Brave", decay_graphs["default"]), default_rp
        ),
    }

    logger.debug(f"Loaded emotion types in {time.perf_counter() - start_time:.02f} s.")

    return emotion_types


def load_emotion_events(resource_dir: str) -> dict[str, EmotionEvent]:

    start_time = time.perf_counter()

    emotion_files = get_json_files(
        resource_dir,
        [os.path.join("cozmo_resources", "config", "engine", "emotionevents/")],
    )
    emotion_events = {}

    for ef in emotion_files:
        json_data = load_json_file(ef)
        if "emotionEvents" not in json_data:
            emotion_events[json_data["name"]] = EmotionEvent.from_json(json_data)
        else:
            for event in json_data["emotionEvents"]:
                emotion_events[event["name"]] = EmotionEvent.from_json(event)

    logger.debug(
        f"Loaded {len(emotion_events)} emotion events in {time.perf_counter() - start_time:.02f} s."
    )

    return emotion_events
