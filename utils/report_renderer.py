from __future__ import annotations

from typing import Any, Dict, List


def _pct(x: float) -> str:
    try:
        return f"{max(0.0, min(1.0, float(x))) * 100:.0f}%"
    except Exception:
        return "-"


def _esc(s: Any) -> str:
    try:
        t = str(s)
    except Exception:
        return ""
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_detection_report_html(report: Dict[str, Any]) -> str:
    """Render a deepfakedetection.io-style report using pure HTML (Gradio-friendly)."""

    overall = report.get("overall_forgery_score", 0.0)
    verdict = report.get("final_verdict", "UNCERTAIN")
    summary = report.get("analysis_summary", "")
    categories: List[Dict[str, Any]] = report.get("categories", []) or []

    overall_pct = _pct(overall)

    badge_color = {
        "AI_GENERATED": "#ef4444",
        "REAL": "#22c55e",
        "UNCERTAIN": "#f59e0b",
    }.get(verdict, "#f59e0b")

    # Basic inline CSS so no external assets are needed.
    parts: List[str] = []
    parts.append(
        """
<style>
*,*::before,*::after{box-sizing:border-box;}
.dd-wrap{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;max-width:980px;margin:0 auto;padding:6px 2px;}

/* Header layout */
.dd-header{display:grid;grid-template-columns:300px 1fr;gap:16px;align-items:stretch;}
@media (max-width: 760px){.dd-header{grid-template-columns:1fr;}}

.dd-score{background:#0b1220;color:#fff;border-radius:14px;padding:18px;min-width:0;}
.dd-score h2{margin:0 0 6px 0;font-size:13px;opacity:.85;font-weight:700;letter-spacing:.2px;}
.dd-score .pct{font-size:44px;line-height:1.0;font-weight:850;margin:6px 0 12px;}
.dd-score .bar{height:10px;background:rgba(255,255,255,.2);border-radius:999px;overflow:hidden;}
.dd-score .bar > div{height:100%;background:#f97316;width:0%;}

.dd-summary{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:18px;min-width:0;}
.dd-summary h2{margin:0 0 10px 0;font-size:13px;color:#111827;opacity:.85;font-weight:800;letter-spacing:.2px;}
.dd-summary .badge{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;color:#fff;font-weight:800;font-size:12px;margin-bottom:10px;}

/* Breakdown */
.dd-breakdown{margin-top:14px;}
.dd-breakdown h2{margin:16px 0 10px;font-size:13px;opacity:.85;font-weight:900;color:#111827;letter-spacing:.2px;}
.dd-card{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:10px 0;background:#fff;}
.dd-card-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px;}
.dd-card-title{font-weight:900;color:#111827;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.dd-card-score{font-weight:900;color:#f97316;flex:0 0 auto;}
.dd-card-bar{height:10px;background:#f3f4f6;border-radius:999px;overflow:hidden;margin:10px 0 8px;}
.dd-card-bar>div{height:100%;background:#f97316;width:0%;}
.dd-card-desc{color:#374151;font-size:13px;line-height:1.4;}
.dd-metrics{margin-top:10px;border-top:1px solid #f3f4f6;padding-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:8px;}
@media (max-width: 760px){.dd-metrics{grid-template-columns:1fr;}}
.dd-metric{background:#f9fafb;border:1px solid #eef2f7;border-radius:10px;padding:10px;min-width:0;}
.dd-metric .k{font-size:11px;color:#6b7280;font-weight:900;letter-spacing:.15px;text-transform:uppercase;margin-bottom:4px;}
.dd-metric .v{font-size:13px;color:#111827;font-weight:800;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.dd-token-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;}
.dd-token{font-size:12px;background:#fff;border:1px solid #e5e7eb;border-radius:999px;padding:3px 8px;color:#111827;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
</style>
"""
    )

    parts.append('<div class="dd-wrap">')
    parts.append('<div class="dd-header">')

    parts.append('<div class="dd-score">')
    parts.append('<h2>Overall Forgery Score</h2>')
    parts.append(f'<div class="pct">{overall_pct}</div>')
    parts.append('<div class="bar"><div style="width:%s"></div></div>' % (overall_pct))
    parts.append('</div>')

    parts.append('<div class="dd-summary">')
    parts.append('<h2>Analysis Summary</h2>')
    parts.append(f'<div class="badge" style="background:{badge_color}">{verdict}</div>')
    parts.append(f'<div style="color:#111827;line-height:1.45;font-size:14px;">{summary}</div>')
    parts.append('</div>')

    parts.append('</div>')

    parts.append('<div class="dd-breakdown">')
    parts.append('<h2>Detailed Breakdown</h2>')

    for cat in categories:
        name = cat.get("name", "Category")
        score = cat.get("score", 0.0)
        desc = cat.get("description", "")
        metrics = cat.get("metrics", {}) or {}
        score_pct = _pct(score)
        parts.append('<div class="dd-card">')
        parts.append('<div class="dd-card-top">')
        parts.append(f'<div class="dd-card-title">{name}</div>')
        parts.append(f'<div class="dd-card-score">{score_pct}</div>')
        parts.append('</div>')
        parts.append('<div class="dd-card-bar"><div style="width:%s"></div></div>' % (score_pct))
        parts.append(f'<div class="dd-card-desc">{desc}</div>')

        if isinstance(metrics, dict) and name.strip().lower() in {"text / symbol artifacts", "text/symbol artifacts", "text artifacts"}:
            text_regions = metrics.get("text_regions")
            mean_conf = metrics.get("mean_conf")
            raw_box_count = metrics.get("raw_box_count")
            sample_tokens = metrics.get("sample_tokens")
            parts.append('<div class="dd-metrics">')
            if raw_box_count is not None:
                parts.append(
                    '<div class="dd-metric"><div class="k">Text Detected</div><div class="v">%s</div></div>'
                    % ("Yes" if int(raw_box_count) > 0 else "No")
                )
            if text_regions is not None:
                parts.append(
                    '<div class="dd-metric"><div class="k">OCR Regions</div><div class="v">%s</div></div>'
                    % (_esc(text_regions))
                )
            if mean_conf is not None:
                try:
                    mean_conf_str = f"{float(mean_conf):.2f}"
                except Exception:
                    mean_conf_str = _esc(mean_conf)
                parts.append(
                    '<div class="dd-metric"><div class="k">Mean OCR Confidence</div><div class="v">%s</div></div>'
                    % (_esc(mean_conf_str))
                )
            if sample_tokens:
                parts.append('<div class="dd-metric" style="grid-column:1/-1;">')
                parts.append('<div class="k">Detected Text (Preview)</div>')
                parts.append('<div class="dd-token-list">')
                try:
                    for tok in list(sample_tokens)[:10]:
                        parts.append(f'<div class="dd-token">{_esc(tok)}</div>')
                except Exception:
                    pass
                parts.append('</div>')
                parts.append('</div>')
            parts.append('</div>')
        parts.append('</div>')

    parts.append('</div>')
    parts.append('</div>')

    return "\n".join(parts)
