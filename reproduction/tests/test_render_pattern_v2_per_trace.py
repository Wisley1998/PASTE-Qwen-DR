from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import render_pattern_v2_per_trace as renderer  # noqa: E402


class PerTraceRendererTests(unittest.TestCase):
    def test_latency_ceiling_preserves_resolution_above_five(self) -> None:
        self.assertEqual(renderer._nice_ceiling(52.8), 60.0)

    def test_json_render_preserves_missing_request_and_accessible_svg(self) -> None:
        payload = {
            "per_request": [
                {
                    "request_number": 1,
                    "trace_id": "trace_task1_example.jsonl",
                    "authoritative_targets": 2,
                    "runtime_target_observations": 8,
                    "top1_target_hits": 1,
                    "top3_target_hits": 1,
                    "top5_target_hits": 2,
                    "runtime_overlap_hits": 1,
                    "runtime_overall_hit_rate": 0.125,
                    "baseline_request_critical_path_proxy_ms_mean": 10.0,
                    "pattern_request_critical_path_proxy_ms_mean": 8.0,
                    "request_critical_path_speedup_ratio": 1.25,
                    "speedup_factor_min": 1.1,
                    "speedup_factor_max": 1.3,
                },
                {
                    "request_number": 3,
                    "trace_id": "trace_task3_example.jsonl",
                    "authoritative_targets": 1,
                    "executable_targets": 1,
                    "top1_recall": 0.0,
                    "top3_recall": 1.0,
                    "top5_recall": 1.0,
                    "runtime_hit_rate": 0.0,
                    "demand_only_latency_ms": 6.0,
                    "pattern_latency_ms": 7.5,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "metrics.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            rows = renderer.complete_request_axis(
                renderer.load_request_metrics(input_path), 3
            )
            self.assertEqual([row.request_number for row in rows], [1, 2, 3])
            self.assertIsNone(rows[1].top1_recall)
            self.assertAlmostEqual(rows[0].speedup_factor, 1.25)

            svg = root / "figure.svg"
            png = root / "figure.png"
            summary = renderer.render(
                rows,
                svg_path=svg,
                png_path=png,
                title="Test chart",
                subtitle="Test protocol",
            )
            svg_text = svg.read_text(encoding="utf-8")
            self.assertIn('aria-labelledby="chart-title chart-desc"', svg_text)
            self.assertIn('<title id="chart-title">Test chart</title>', svg_text)
            self.assertIn('<desc id="chart-desc">', svg_text)
            self.assertIn('id="na-hatch"', svg_text)
            self.assertIn("Request number", svg_text)
            self.assertEqual(summary["top1"]["text"], "1/3  33.3%")
            self.assertEqual(summary["runtime"]["text"], "1/9  11.1%")
            self.assertAlmostEqual(
                summary["latency"]["weighted_speedup_factor"], 16.0 / 15.5
            )
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 1000)

            from PIL import Image

            with Image.open(png) as image:
                self.assertEqual(image.size, (renderer.WIDTH, renderer.HEIGHT))
                self.assertEqual(image.mode, "RGB")

    def test_runtime_denominator_displays_replay_multiplier(self) -> None:
        rows = [
            renderer.parse_request_metric(
                {
                    "request_number": 1,
                    "authoritative_targets": 2,
                    "runtime_target_observations": 16,
                    "top1_target_hits": 1,
                    "top3_target_hits": 1,
                    "top5_target_hits": 1,
                    "runtime_overlap_hits": 2,
                    "runtime_overall_hit_rate": 0.125,
                }
            )
        ]

        summary = renderer.aggregate_summary(rows)

        self.assertEqual(summary["runtime"]["text"], "2/(2x8)  12.5%")

    def test_rejects_nonmonotone_topk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "request_number": 1,
                            "authoritative_targets": 1,
                            "top1_recall": 1.0,
                            "top3_recall": 0.0,
                            "top5_recall": 1.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not monotone"):
                renderer.load_request_metrics(path)


if __name__ == "__main__":
    unittest.main()
