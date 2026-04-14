import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .parser import parse_prediction
except ImportError:
    from parser import parse_prediction


SYSTEM_PROMPT = """You are a strict recovery-action predictor for embodied failure cases.
Return a JSON object only, with keys:
- recovery_action (required): one action string from closed action space
- failure_type (optional): one of F1,F2,F3,F4,F5
- evidence (optional): short phrase
- reasoning (optional): short reasoning for qualitative analysis
Do not output markdown or extra text."""

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _load_system_prompt(strategy: str) -> str:
    """Load system prompt based on reasoning strategy."""
    if strategy == "vanilla":
        return SYSTEM_PROMPT

    prompt_file = PROMPT_DIR / f"{strategy}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        f"Prompt file not found: {prompt_file}\n"
        f"Available: {[f.stem for f in PROMPT_DIR.glob('*.txt')]}"
    )


def _load_dotenv_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _maybe_load_env(project_root: Path) -> None:
    dotenv_path = project_root / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path, override=False)
    except Exception:
        _load_dotenv_file(dotenv_path)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_done_ids(output_path: Path, mode: str, model: str) -> set:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("mode") == mode and row.get("model") == model:
                done.add(row.get("sample_id"))
    return done


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def _to_data_url(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{_guess_mime(path)};base64,{b64}"


def _user_text(sample: Dict, strategy: str = "vanilla") -> str:
    """Build user message text. Adjusts JSON schema hint based on strategy."""
    base = (
        f"Task instruction: {sample['instruction']}\n"
        f"Attempted action: {sample['attempted_action']}\n"
        "Closed action space: LookAround(), Navigate(target), Pick(obj), Place(obj,loc), "
        "Open(container), Close(container), Retry(action)\n"
        f"Candidate vocab: {json.dumps(sample['candidate_vocab'], ensure_ascii=True)}\n"
    )

    if strategy == "vanilla":
        base += (
            "Output JSON schema:\n"
            '{"recovery_action":"...", "failure_type":"F1|F2|F3|F4|F5", '
            '"evidence":"...", "reasoning":"..."}\n'
            "Return JSON only."
        )
    else:
        base += (
            'Your JSON output MUST include "recovery_action" as a required key.\n'
            "Choose recovery_action ONLY from the closed action space and candidate vocab above.\n"
            "Return JSON only."
        )

    return base


def _build_content_parts(
    sample: Dict, mode: str, project_root: Path, strategy: str = "vanilla"
) -> List[Dict]:
    def resolve_image_path(rel: str) -> Path:
        p1 = (project_root / rel).resolve()
        if p1.exists():
            return p1
        p2 = (project_root / "data" / rel.replace("./", "")).resolve()
        if p2.exists():
            return p2
        return p1

    before_path = resolve_image_path(sample["before_image"])
    after_path = resolve_image_path(sample["after_image"])
    parts = [{"type": "text", "text": _user_text(sample, strategy)}]

    if mode == "text-only":
        return parts

    if mode == "after-only":
        after_data = _to_data_url(after_path)
        order = [("image_1", after_data, after_path)]
    elif mode == "swap-test":
        before_data = _to_data_url(after_path)
        after_data = _to_data_url(before_path)
        order = [("before", before_data, after_path), ("after", after_data, before_path)]
    else:
        before_data = _to_data_url(before_path)
        after_data = _to_data_url(after_path)
        order = [("before", before_data, before_path), ("after", after_data, after_path)]

    for label, data_url, local_path in order:
        if data_url:
            parts.append({"type": "text", "text": f"{label} image"})
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            parts.append({"type": "text", "text": f"{label} image missing at {local_path}"})
    return parts


def _extract_usage(response: Dict) -> Tuple[int, int]:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return prompt_tokens, completion_tokens


def _default_base_url(model: str, user_base_url: Optional[str]) -> str:
    if user_base_url:
        return user_base_url.rstrip("/")
    if "qwen" in model.lower():
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://api.openai.com/v1"


def _api_key_for_model(model: str) -> Optional[str]:
    if "qwen" in model.lower():
        return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def _request_chat_completion(
    *,
    model: str,
    base_url: str,
    api_key: str,
    messages: List[Dict],
    timeout_sec: int = 120,
) -> Dict:
    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_text(response: Dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                texts.append(part.get("text", ""))
        return "\n".join(texts).strip()
    return ""


def _append_jsonl(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _extract_reasoning_payload(text: str) -> Dict:
    """Extract ALL reasoning fields from model response.

    Returns dict with:
      - reasoning_text: primary text (first non-empty field)
      - source_field: which field was used as primary
      - all_reasoning_fields: dict of ALL non-empty reasoning fields
    """
    payload = {
        "reasoning_text": "",
        "source_field": "none",
        "all_reasoning_fields": {},
    }
    try:
        obj = json.loads(text)
    except Exception:
        return payload
    if not isinstance(obj, dict):
        return payload

    strategy_fields = [
        "discrepancy",
        "failure_point",
        "reasoning",
        "evidence",
        "expected_state",
        "observed_state",
        "preconditions",
        "execution_trace",
    ]

    all_fields = {}
    primary_text = ""
    primary_field = "none"

    for field in strategy_fields:
        val = obj.get(field)
        if isinstance(val, str) and val.strip():
            all_fields[field] = val.strip()
            if not primary_text:
                primary_text = val.strip()
                primary_field = field
        elif isinstance(val, list) and val:
            joined = "; ".join(str(v) for v in val)
            all_fields[field] = joined
            if not primary_text:
                primary_text = joined
                primary_field = field

    payload["reasoning_text"] = primary_text
    payload["source_field"] = primary_field
    payload["all_reasoning_fields"] = all_fields
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM predictor with incremental saving, ablations, and reasoning strategies"
    )
    parser.add_argument("--data", default="data/samples.jsonl", help="input samples jsonl")
    parser.add_argument("--output", default="outputs/predictions.jsonl", help="output predictions jsonl")
    parser.add_argument("--project-root", default=".", help="project root for image paths")
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="model name (e.g., gpt-4o-mini, gpt-5.4, qwen2-vl-72b-instruct)"
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument(
        "--mode", default="full",
        choices=["full", "after-only", "swap-test", "text-only"]
    )
    parser.add_argument(
        "--prompt-strategy", default="vanilla",
        choices=["vanilla", "counterfactual", "mental_simulation"],
        help="reasoning strategy for system prompt"
    )
    parser.add_argument("--start", type=int, default=0, help="start index")
    parser.add_argument("--limit", type=int, default=0, help="max samples (0 = all)")
    parser.add_argument("--sleep-ms", type=int, default=0, help="delay between calls (ms)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output file")
    parser.add_argument("--logs-dir", default="data/logs", help="directory for reasoning logs")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    out_path = Path(args.output).resolve()
    project_root = Path(args.project_root).resolve()
    logs_dir = (project_root / args.logs_dir).resolve()
    reason_log_path = logs_dir / f"reasoning_{args.model}_{args.mode}_{args.prompt_strategy}.jsonl"
    _maybe_load_env(project_root)
    samples = load_jsonl(data_path)

    # Load system prompt based on strategy
    system_prompt = _load_system_prompt(args.prompt_strategy)
    print(f"Prompt strategy: {args.prompt_strategy}")
    print(f"System prompt length: {len(system_prompt)} chars")

    if args.overwrite and out_path.exists():
        out_path.unlink()
    if args.overwrite and reason_log_path.exists():
        reason_log_path.unlink()

    base_url = _default_base_url(args.model, args.base_url)
    api_key = _api_key_for_model(args.model)
    if not api_key:
        raise RuntimeError(
            "Missing API key: set OPENAI_API_KEY (gpt) or QWEN_API_KEY/DASHSCOPE_API_KEY (qwen)."
        )

    done_ids = _read_done_ids(out_path, args.mode, args.model)
    end = len(samples) if args.limit <= 0 else min(len(samples), args.start + args.limit)
    sliced = samples[args.start:end]
    print(
        f"Loaded {len(samples)} samples, processing [{args.start}:{end}) "
        f"mode={args.mode}, model={args.model}, strategy={args.prompt_strategy}"
    )
    print(f"Resume: {len(done_ids)} already completed")

    for idx, sample in enumerate(sliced, start=args.start):
        sample_id = sample["sample_id"]
        if sample_id in done_ids:
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_content_parts(
                    sample, args.mode, project_root, args.prompt_strategy
                ),
            },
        ]

        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        row = {
            "sample_id": sample_id,
            "index": idx,
            "mode": args.mode,
            "model": args.model,
            "prompt_strategy": args.prompt_strategy,
            "timestamp": ts,
            "status": "ok",
        }

        try:
            response = _request_chat_completion(
                model=args.model,
                base_url=base_url,
                api_key=api_key,
                messages=messages,
            )
            text = _extract_text(response)
            prompt_tokens, completion_tokens = _extract_usage(response)
            parsed = parse_prediction(text, sample["candidate_vocab"])
            reason_payload = _extract_reasoning_payload(text)
            row["raw_response_text"] = text
            row["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            row["parsed"] = {
                "valid_json": parsed.valid_json,
                "valid_action": parsed.valid_action,
                "recovery_action": parsed.recovery_action,
                "failure_type": parsed.failure_type,
                "evidence": parsed.evidence,
                "error": parsed.error,
            }
            reason_row = {
                "sample_id": sample_id,
                "index": idx,
                "mode": args.mode,
                "model": args.model,
                "prompt_strategy": args.prompt_strategy,
                "timestamp": ts,
                "status": row["status"],
                "reasoning_text": reason_payload["reasoning_text"],
                "reasoning_source_field": reason_payload["source_field"],
                "all_reasoning_fields": reason_payload["all_reasoning_fields"],
                "raw_response_text": text,
                "parsed_recovery_action": parsed.recovery_action,
                "parsed_failure_type": parsed.failure_type,
            }
            _append_jsonl(reason_log_path, reason_row)
        except urllib.error.HTTPError as exc:
            row["status"] = "http_error"
            row["error"] = f"{exc.code} {exc.reason}"
            try:
                row["error_body"] = exc.read().decode("utf-8", errors="replace")
            except Exception:
                row["error_body"] = ""
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)

        _append_jsonl(out_path, row)
        done_ids.add(sample_id)
        print(f"[{idx+1}/{end}] {sample_id} -> {row['status']}")
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"Done. Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()