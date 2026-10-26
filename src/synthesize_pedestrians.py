#!/usr/bin/env python3
"""Create deterministic LiDAR pedestrians on synthetic UrbanTwin backgrounds."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import read_detection_labels, read_points, write_detection_labels


def sample_ellipsoid(rng: np.random.Generator, center: np.ndarray, radii: np.ndarray, count: int) -> np.ndarray:
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-6)
    return center + direction * radii


def person_surface(rng: np.random.Generator, height: float, width: float, depth: float, count: int = 2400) -> np.ndarray:
    # Head, torso and two legs. Surface oversampling is reduced by angular
    # quantization below to approximate returns from a rotating roadside LiDAR.
    n_head = count // 6
    n_torso = count // 2
    n_leg = (count - n_head - n_torso) // 2
    parts = [
        sample_ellipsoid(rng, np.array([0, 0, 0.88 * height]), np.array([0.17, 0.15, 0.12]) * height, n_head),
        sample_ellipsoid(rng, np.array([0, 0, 0.58 * height]), np.array([0.50 * width, 0.50 * depth, 0.25 * height]), n_torso),
        sample_ellipsoid(rng, np.array([0.20 * width, 0, 0.23 * height]), np.array([0.16, 0.18, 0.25]) * np.array([width, depth, height]), n_leg),
        sample_ellipsoid(rng, np.array([-0.20 * width, 0, 0.23 * height]), np.array([0.16, 0.18, 0.25]) * np.array([width, depth, height]), count - n_head - n_torso - n_leg),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def lidar_visible(points: np.ndarray, horizontal_deg: float = 0.18, vertical_deg: float = 0.35) -> np.ndarray:
    radius = np.linalg.norm(points, axis=1).clip(1e-6)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    elevation = np.arcsin(np.clip(points[:, 2] / radius, -1, 1))
    bins = np.column_stack((
        np.round(np.rad2deg(azimuth) / horizontal_deg),
        np.round(np.rad2deg(elevation) / vertical_deg),
    )).astype(np.int64)
    order = np.argsort(radius)
    _, first = np.unique(bins[order], axis=0, return_index=True)
    return points[order[np.sort(first)]]


def overlaps_xy(box: np.ndarray, boxes: list[np.ndarray], margin: float = 0.8) -> bool:
    for other in boxes:
        if abs(box[0] - other[0]) < (box[3] + other[3]) / 2 + margin and abs(box[1] - other[1]) < (box[4] + other[4]) / 2 + margin:
            return True
    return False


def local_ground_z(points: np.ndarray, x: float, y: float, radius: float = 1.5) -> float | None:
    distance2 = (points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2
    nearby = points[distance2 < radius**2, 2]
    if len(nearby) < 20:
        return None
    return float(np.percentile(nearby, 12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment synthetic frames with procedural pedestrian LiDAR returns")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--max-source-frames", type=int, default=8000)
    parser.add_argument("--people-min", type=int, default=1)
    parser.add_argument("--people-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    train_file = args.data / "ImageSets" / "train.txt"
    source_ids = train_file.read_text().split()[: args.max_source_frames]
    next_id = max(int(x) for x in source_ids) + 1
    augmented_ids = []

    for variant in range(args.variants):
        for source_id in source_ids:
            rng = np.random.default_rng(args.seed + variant * 1_000_003 + int(source_id))
            points = read_points(args.data / "points" / f"{source_id}.npy")
            labels = read_detection_labels(args.data / "labels" / f"{source_id}.txt")
            boxes = [box for box, _ in labels]
            people = []
            target_count = int(rng.integers(args.people_min, args.people_max + 1))
            attempts = 0
            while len(people) < target_count and attempts < 100:
                attempts += 1
                distance = float(rng.uniform(7, 38))
                angle = float(rng.uniform(-math.pi, math.pi))
                x, y = distance * math.cos(angle), distance * math.sin(angle)
                ground = local_ground_z(points, x, y)
                if ground is None or not (-8 < ground < 1):
                    continue
                height = float(np.clip(rng.normal(1.70, 0.12), 1.40, 2.0))
                width = float(np.clip(rng.normal(0.62, 0.07), 0.45, 0.82))
                depth = float(np.clip(rng.normal(0.55, 0.08), 0.40, 0.80))
                heading = float(rng.uniform(-math.pi, math.pi))
                box = np.array([x, y, ground + height / 2, width, depth, height, heading], dtype=np.float32)
                if overlaps_xy(box, boxes + [b for b, _ in people]):
                    continue

                surface = person_surface(rng, height, width, depth)
                c, s = math.cos(heading), math.sin(heading)
                surface[:, :2] = surface[:, :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)
                surface += np.array([x, y, ground], dtype=np.float32)
                visible = lidar_visible(surface)
                if len(visible) < 3:
                    continue
                people.append((box, visible))

            if not people:
                continue
            xyz = np.concatenate([p for _, p in people], axis=0)
            intensity = np.clip(rng.normal(0.35, 0.12, size=(len(xyz), 1)), 0, 1).astype(np.float32)
            augmented = np.concatenate((points, np.concatenate((xyz, intensity), axis=1)), axis=0)
            output_id = f"{next_id:06d}"
            next_id += 1
            np.save(args.data / "points" / f"{output_id}.npy", augmented)
            write_detection_labels(
                args.data / "labels" / f"{output_id}.txt",
                labels + [(box, "pedestrian") for box, _ in people],
            )
            augmented_ids.append(output_id)

    with train_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(augmented_ids) + ("\n" if augmented_ids else ""))
    print(f"Created {len(augmented_ids)} augmented frames with procedural pedestrians")


if __name__ == "__main__":
    main()

