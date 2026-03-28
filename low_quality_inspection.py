import json, re, hashlib
from typing import List, Dict, Any, Tuple
import numpy as np

# -----------------------------
# 1) 规则/统计特征（快、便宜）
# -----------------------------

BAD_MARKERS = [
    r"\[\!\[Image.*?\]\(",          # Markdown 图片占位
    r"\( ?On the other hand",       # 截断/模板残片（只是示例）
    r"Click here", r"Subscribe",    # 导流模板
    r"cookie", r"privacy policy"
]

def safe_float_pct(x: str) -> float:
    x = (x or "").strip().replace("+", "")
    if "±" in x:
        return 0.0
    try:
        return float(x)
    except:
        return 0.0

def text_stats(s: str) -> Dict[str, float]:
    s = s or ""
    n = len(s)
    if n == 0:
        return {"len": 0, "alpha_ratio": 0.0, "punct_ratio": 0.0, "digit_ratio": 0.0, "uniq_ratio": 0.0}

    alpha = sum(ch.isalpha() for ch in s)
    digit = sum(ch.isdigit() for ch in s)
    punct = sum(ch in ".,;:!?()[]{}<>/\\|@#$%^&*~`" for ch in s)

    tokens = re.findall(r"\w+", s.lower())
    uniq_ratio = len(set(tokens)) / max(1, len(tokens))

    return {
        "len": n,
        "alpha_ratio": alpha / n,
        "digit_ratio": digit / n,
        "punct_ratio": punct / n,
        "uniq_ratio": uniq_ratio
    }

def rule_quality_score(item: Dict[str, Any]) -> Tuple[float, List[str]]:
    reasons = []
    full = item.get("full_article", "") or ""
    st = text_stats(full)

    score = 1.0

    if st["len"] < 800:
        score -= 0.35
        reasons.append("too_short")

    if st["uniq_ratio"] < 0.25:
        score -= 0.25
        reasons.append("low_unique_token_ratio")

    if st["alpha_ratio"] < 0.55:
        score -= 0.20
        reasons.append("low_alpha_ratio")

    bad_hit = 0
    for pat in BAD_MARKERS:
        if re.search(pat, full, flags=re.IGNORECASE):
            bad_hit += 1
    if bad_hit >= 2:
        score -= 0.30
        reasons.append("many_bad_markers")

    imp = item.get("impact_analysis", {}) or {}

    def check(node_key: str) -> bool:
        node = imp.get(node_key, {}) or {}
        eff = (node.get("effect") or "").lower()
        pct = safe_float_pct(node.get("percentage_change", "0"))
        if eff == "rise" and pct < 0:
            return False
        if eff == "fall" and pct > 0:
            return False
        return True

    ok = True
    ok = ok and check("short_term_impact_5_days")
    ok = ok and check("medium_term_impact_15_days")
    ok = ok and check("long_term_impact_after_15_days")
    if not ok:
        score -= 0.25
        reasons.append("impact_inconsistent")

    rel = float(item.get("relevance_to_bitcoin_price", 0.0) or 0.0)
    if rel < 0.4:
        score -= 0.20
        reasons.append("low_relevance")

    score = float(np.clip(score, 0.0, 1.0))
    return score, reasons

# -----------------------------
# 2) 近重复去重（事件级）目前只能处理英文。基于北bm25 embedding余玄计算速度比较慢，用这个hash的速度会快很多。
# -----------------------------

def normalize_for_hash(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()

def simhash64(tokens: List[str]) -> int:
    v = [0] * 64
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def dedup_group(items: List[Dict[str, Any]], ham_thr: int = 6) -> List[int]:
    sigs = []
    for it in items:
        text = normalize_for_hash((it.get("title", "") or "") + " " + (it.get("full_article", "") or "")[:2000])
        toks = text.split()
        sigs.append(simhash64(toks[:256]))

    group_id = [-1] * len(items)
    gid = 0
    for i in range(len(items)):
        if group_id[i] != -1:
            continue
        group_id[i] = gid
        for j in range(i + 1, len(items)):
            if group_id[j] != -1:
                continue
            if hamming(sigs[i], sigs[j]) <= ham_thr:
                group_id[j] = gid
        gid += 1
    return group_id

# -----------------------------
# 3) LLM Judge（可选）
# -----------------------------

JUDGE_PROMPT = """You are a data quality auditor for crypto-market news.
Score whether this article is (A) informative & Bitcoin-causal, (B) clean & non-template, (C) not duplicated/rewritten.
Return strict JSON:
{"is_good": true/false,
 "reasons": [..],
 "quality_score": 0.0-1.0,
 "btc_causality_score": 0.0-1.0,
 "template_score": 0.0-1.0}
"""

def build_judge_user_prompt(item: Dict[str, Any]) -> str:
    return (
        f"TITLE: {item.get('title','')}\n"
        f"TIME: {item.get('publication_time','')}\n"
        f"LINK: {item.get('link','')}\n"
        f"RELEVANCE: {item.get('relevance_to_bitcoin_price','')}\n"
        f"IMPACT: {json.dumps(item.get('impact_analysis',{}), ensure_ascii=False)}\n"
        f"ARTICLE:\n{(item.get('full_article','') or '')[:6000]}\n"
    )

# -----------------------------
# 配置：不用 dataclass
# -----------------------------

def default_config():
    return {
        "min_rule_score": 0.55,
        "keep_top1_per_dedup_group": True,
        "use_llm_judge": False,
        "llm_votes": 3,
        "llm_keep_threshold": 0.6,
        "llm_min_causality": 0.5,
        "llm_min_template": 0.5
    }

# -----------------------------
# 主入口
# -----------------------------

def filter_dataset(input_path: str, output_path: str, cfg: Dict[str, Any], agent=None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 规则初筛
    kept = []
    dropped = []
    for it in data:
        score, reasons = rule_quality_score(it)
        it["_rule_quality_score"] = score
        it["_rule_drop_reasons"] = reasons
        if score >= float(cfg.get("min_rule_score", 0.55)):
            kept.append(it)
        else:
            it["_drop_reason"] = "rule_filter"
            dropped.append(it)

    # 2) 去重
    gids = dedup_group(kept)
    for it, gid in zip(kept, gids):
        it["_dedup_group_id"] = int(gid)

    kept2 = kept
    if bool(cfg.get("keep_top1_per_dedup_group", True)):
        best = {}
        for it in kept2:
            gid = it["_dedup_group_id"]
            if gid not in best or it["_rule_quality_score"] > best[gid]["_rule_quality_score"]:
                best[gid] = it
        kept2 = list(best.values())

    # 3) 可选：LLM Judge
    if bool(cfg.get("use_llm_judge", False)):
        if agent is None:
            raise ValueError("use_llm_judge=True 时必须传入 agent（例如你的 QwenVllmAgent 实例）")

        prompts = [build_judge_user_prompt(it) for it in kept2]

        votes = []
        llm_votes = int(cfg.get("llm_votes", 3))
        agent.set_system_prompt(JUDGE_PROMPT)
        for _ in range(llm_votes):
            # 这里假设 agent.query 返回 list[dict|None]
            votes.append(agent.query(prompts, plain=False, print_prompt=False))

        final_kept = []
        for i, it in enumerate(kept2):
            js = []
            for v in votes:
                if i < len(v) and isinstance(v[i], dict):
                    js.append(v[i])

            if not js:
                it["_drop_reason"] = "llm_judge_parse_fail"
                dropped.append(it)
                continue

            avg_q = float(np.mean([x.get("quality_score", 0.0) for x in js]))
            avg_c = float(np.mean([x.get("btc_causality_score", 0.0) for x in js]))
            avg_t = float(np.mean([x.get("template_score", 0.0) for x in js]))

            it["_llm_quality_score"] = avg_q
            it["_llm_btc_causality"] = avg_c
            it["_llm_template_score"] = avg_t
            it["_llm_votes"] = js

            if (avg_q >= float(cfg.get("llm_keep_threshold", 0.6)) and
                avg_c >= float(cfg.get("llm_min_causality", 0.5)) and
                avg_t >= float(cfg.get("llm_min_template", 0.5))):
                final_kept.append(it)
            else:
                it["_drop_reason"] = "llm_judge_filter"
                dropped.append(it)

        kept2 = final_kept

    out = {
        "kept": kept2,
        "dropped": dropped,
        "stats": {
            "raw": len(data),
            "kept_after_rule": len(kept),
            "kept_after_dedup": len(kept2),
            "dropped": len(dropped),
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Done:", out["stats"])

from Constant import *
import numpy as np
import torch
from QwenAgent import QwenAgent
from QwenAgent_vllm import QwenVllmAgent
if __name__ == "__main__":
    cfg = default_config()
    # 先不开 LLM Judge：先跑规则+去重
    cfg["use_llm_judge"] = True
    agent = QwenVllmAgent(
        model_name=QWEN_2_5_AGENT_72B_NAME,
        system_prompt=JUDGE_PROMPT,
        device_parallel=4,
        is_api=False,
        max_new_tokens=512
    )

    filter_dataset(
        input_path=r"/scratch/u5ci/liuhongyuan.u5ci/timellm/time_agent/bitcoin_news_analysis_results_full_processed.json",
        output_path=r"/scratch/u5ci/liuhongyuan.u5ci/timellm/time_agent/bitcoin_news_analysis_results_filtered-72b.json",
        cfg=cfg,
        agent=agent
    )
