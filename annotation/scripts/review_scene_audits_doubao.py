#!/usr/bin/env python3
"""Final review and arbitration for scene audit results using the Doubao API."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from scene_audit_prompts import SCENE_REVIEW_PROMPT, SCENE_REVIEW_USER_INTRO
from scene_audit_utils import encode_image_data_url, ensure_dir, eprint, list_images, list_scenes, save_json


PROGRAMMATIC_FIELDS = [
    "brightness_level",
    "illumination_level",
    "dynamic_range_level",
    "highlight_strength",
    "dark_region_ratio_level",
]

TARGET_FIELDS = [
    "scene_category",
    "layout_plausibility",
    "functional_coherence",
    "object_arrangement_coherence",
    "lighting_naturalness",
    "brightness_level",
    "illumination_level",
    "dynamic_range_level",
    "highlight_strength",
    "shadow_strength",
    "dark_region_ratio_level",
    "lighting_challenge_type",
    "overall_plausibility",
    "confidence",
    "issues",
    "reason_short",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_label_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def detect_disagreement_fields(merged: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    for key in TARGET_FIELDS:
        if key not in merged or not isinstance(merged[key], list):
            continue
        unique_values = []
        for item in merged[key]:
            normalized = normalize_label_value(item)
            if normalized not in unique_values:
                unique_values.append(normalized)
        if len(unique_values) > 1:
            fields.append(key)
    return fields


def build_review_context(merged: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "scene": merged.get("scene"),
        "images": merged.get("images"),
        "disagreement_fields": detect_disagreement_fields(merged),
        "merged_fields": {key: merged[key] for key in TARGET_FIELDS if key in merged},
        "computed": merged.get("computed", {}),
        "parse_error": merged.get("parse_error"),
    }
    if "raw" in merged:
        context["model_outputs"] = merged["raw"]
    return context


def build_review_messages(image_paths: List[str], merged_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    context_text = json.dumps(build_review_context(merged_json), ensure_ascii=False, indent=2)
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{SCENE_REVIEW_USER_INTRO}\n\nMerged review context:\n{context_text}",
        }
    ]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(path)}})
    return [
        {"role": "system", "content": SCENE_REVIEW_PROMPT},
        {"role": "user", "content": content},
    ]


def extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "empty_response"
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed, None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None, "json_not_found"
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed, None
    except Exception as exc:
        return None, f"json_parse_error: {exc}"
    return None, "json_not_found"


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
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
    return response.json()


def preserve_programmatic_fields(final_obj: Dict[str, Any], merged: Dict[str, Any]) -> None:
    for key in PROGRAMMATIC_FIELDS:
        if key in merged:
            final_obj[key] = merged[key]


def validate_minimum_fields(final_obj: Dict[str, Any]) -> Dict[str, Any]:
    for key in TARGET_FIELDS:
        if key not in final_obj:
            final_obj[key] = [] if key == "issues" else 0.0 if key == "confidence" else "uncertain"
    return final_obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Final review and arbitration with the Doubao API")
    parser.add_argument("--input", required=True, help="Scene directory with one subdirectory per scene.")
    parser.add_argument("--merged", required=True, help="Directory containing merged scene-audit JSON files.")
    parser.add_argument("--output", required=True, help="Output directory for reviewed JSON files.")
    parser.add_argument("--scene", default=None, help="Only process the specified scene name.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of scenes to process. Use 0 for all scenes.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum number of images per scene. Use 0 for all available images.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep time in seconds after each API call.")
    parser.add_argument("--model", default=os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215"), help="Model name.")
    parser.add_argument("--base-url", default=os.getenv("DOUBAO_BASE_URL", os.getenv("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")), help="API base URL.")
    parser.add_argument("--api-key", default=os.getenv("ARK_API_KEY") or os.getenv("DOUBAO_API_KEY") or os.getenv("VOLCENGINE_API_KEY"), help="API key.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if not args.api_key:
        eprint("No API key found. Set ARK_API_KEY, DOUBAO_API_KEY, or VOLCENGINE_API_KEY.")
        return 2
    if not os.path.isdir(args.input):
        eprint(f"Scene directory does not exist: {args.input}")
        return 2
    if not os.path.isdir(args.merged):
        eprint(f"Merged JSON directory does not exist: {args.merged}")
        return 2

    ensure_dir(args.output)
    scenes = list_scenes(args.input)
    if args.scene:
        scenes = [scene for scene in scenes if scene == args.scene]
    if args.limit > 0:
        scenes = scenes[: args.limit]
    if not scenes:
        eprint("No scenes found to process.")
        return 1

    results_path = os.path.join(args.output, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as handle:
        for index, scene in enumerate(scenes, 1):
            scene_dir = os.path.join(args.input, scene)
            merged_path = os.path.join(args.merged, f"{scene}.json")
            if not os.path.isfile(merged_path):
                eprint(f"[{scene}] merged JSON not found; skipping: {merged_path}")
                continue

            image_paths = list_images(scene_dir)
            if args.max_images > 0:
                image_paths = image_paths[: args.max_images]
            if not image_paths:
                eprint(f"[{scene}] no matching images found; skipping")
                continue

            try:
                merged_obj = load_json(merged_path)
            except Exception as exc:
                eprint(f"[{scene}] failed to read merged JSON: {exc}")
                continue

            eprint(f"[{index}/{len(scenes)}] reviewing scene: {scene} (images={len(image_paths)})")
            try:
                response = post_chat_completion(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=build_review_messages(image_paths, merged_obj),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                raw_content = response["choices"][0]["message"]["content"]
            except Exception as exc:
                eprint(f"[{scene}] request failed: {exc}")
                failure = {"scene": scene, "images": [os.path.basename(path) for path in image_paths], "merged_json": merged_path, "error": str(exc)}
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                continue

            parsed, parse_error = extract_json(raw_content or "")
            if isinstance(parsed, dict):
                preserve_programmatic_fields(parsed, merged_obj)
                parsed = validate_minimum_fields(parsed)

            record = {
                "scene": scene,
                "images": [os.path.basename(path) for path in image_paths],
                "merged_json": merged_path,
                "review_raw": raw_content,
                "review_parsed": parsed,
                "parse_error": parse_error,
                "disagreement_fields": detect_disagreement_fields(merged_obj),
                "computed": merged_obj.get("computed", {}),
            }
            save_json(os.path.join(args.output, f"{scene}.json"), record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.sleep > 0:
                time.sleep(args.sleep)

    eprint(f"Done. Reviewed results written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
