#!/usr/bin/env python3
"""Batch scene audit with the Qwen API and programmatic lighting statistics."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Set

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from scene_audit_prompts import SCENE_AUDIT_PROMPT, SCENE_AUDIT_USER_TEXT
from scene_audit_utils import (
    apply_lighting_buckets,
    compute_lighting_stats,
    encode_image_data_url,
    ensure_dir,
    eprint,
    extract_json,
    list_images,
    list_scenes,
    save_json,
)


def build_messages(image_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": SCENE_AUDIT_USER_TEXT}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(path)}})
    return [
        {"role": "system", "content": SCENE_AUDIT_PROMPT},
        {"role": "user", "content": content},
    ]


def post_chat_completion(base_url: str, api_key: str, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int, timeout: int) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed; HTTP requests are unavailable.")
    url = base_url if base_url.endswith("/chat/completions") else base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}),
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def list_completed_scenes(output_dir: str) -> Set[str]:
    completed: Set[str] = set()
    if not os.path.isdir(output_dir):
        return completed
    for name in os.listdir(output_dir):
        if name.endswith(".json") and name != "results.json":
            completed.add(os.path.splitext(name)[0])
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch scene audit with the Qwen API")
    parser.add_argument("--input", default="G:\\interiorverse\\part_0_png", help="Input directory with one subdirectory per scene.")
    parser.add_argument("--output", default="G:\\interiorverse\\part_0_png_json", help="Output directory for per-scene JSON files.")
    parser.add_argument("--scene", default=None, help="Only process the specified scene name.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of scenes to process. Use 0 for all scenes.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep time in seconds after each API call.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum number of images per scene. Use 0 for all available images.")
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "qwen3-vl-plus"), help="Model name.")
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), help="API base URL.")
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"), help="API key.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if not args.api_key:
        eprint("No API key found. Set DASHSCOPE_API_KEY or QWEN_API_KEY.")
        return 2
    if not os.path.isdir(args.input):
        eprint(f"Input directory does not exist: {args.input}")
        return 2

    ensure_dir(args.output)
    scenes = list_scenes(args.input)
    if args.scene:
        scenes = [scene for scene in scenes if scene == args.scene]
    total_scenes = len(scenes)
    completed_scenes = list_completed_scenes(args.output)
    scenes = [scene for scene in scenes if scene not in completed_scenes]
    skipped_scenes = total_scenes - len(scenes)
    if args.limit > 0:
        scenes = scenes[: args.limit]
    if not scenes:
        eprint("No scenes found to process.")
        return 1

    eprint(f"Scene summary: total={total_scenes}, skipped={skipped_scenes}, remaining={len(scenes)}")
    results_path = os.path.join(args.output, "results.jsonl")
    with open(results_path, "a", encoding="utf-8") as handle:
        for index, scene in enumerate(scenes, 1):
            scene_dir = os.path.join(args.input, scene)
            image_paths = list_images(scene_dir)
            if args.max_images > 0:
                image_paths = image_paths[: args.max_images]
            if not image_paths:
                eprint(f"[{scene}] no matching images found; skipping")
                continue

            eprint(f"[{index}/{len(scenes)}] processing scene: {scene} (images={len(image_paths)})")
            try:
                response = post_chat_completion(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=build_messages(image_paths),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                raw_content = response["choices"][0]["message"]["content"]
            except Exception as exc:
                eprint(f"[{scene}] request failed: {exc}")
                failure = {"scene": scene, "images": [os.path.basename(path) for path in image_paths], "error": str(exc)}
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                continue

            parsed, parse_error = extract_json(raw_content or "")
            stats = compute_lighting_stats(image_paths)
            if isinstance(parsed, dict):
                apply_lighting_buckets(parsed, stats)

            record: Dict[str, Any] = {
                "scene": scene,
                "images": [os.path.basename(path) for path in image_paths],
                "raw": raw_content,
                "parsed": parsed,
                "parse_error": parse_error,
            }
            if stats:
                record["computed"] = stats

            save_json(os.path.join(args.output, f"{scene}.json"), record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.sleep > 0:
                time.sleep(args.sleep)

    eprint(f"Done. Results written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
