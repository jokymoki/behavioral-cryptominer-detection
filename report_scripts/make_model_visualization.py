from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from project_config import H, T

OUT_PATH = BASE_DIR / "figures" / "report" / "tcn_model_detailed_visualization.svg"


FEATURES_USED = 26
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]
RECEPTIVE_FIELD = 1 + 2 * (KERNEL_SIZE - 1) * sum(DILATIONS)


def rect(x, y, w, h, fill, stroke="#263238", radius=8):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    )


def text(x, y, value, size=16, weight="400", fill="#172026", anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{value}</text>'
    )


def line(x1, y1, x2, y2, stroke="#455a64", width=2.0, marker=True, dash=None):
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}{marker_attr}/>'
    )


def block(x, y, w, h, title, subtitle, fill, accent="#263238"):
    cx = x + w / 2
    return "\n".join(
        [
            rect(x, y, w, h, fill),
            text(cx, y + 30, title, size=17, weight="700", fill=accent),
            text(cx, y + 58, subtitle, size=13, fill="#37474f"),
        ]
    )


def build_svg():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M2,2 L10,6 L2,10 Z" fill="#455a64"/>',
        "</marker>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#000000" flood-opacity="0.14"/>',
        "</filter>",
        "</defs>",
        '<rect width="1600" height="900" fill="#f7f9fb"/>',
        text(800, 56, "Detailed TCN Forecasting Model", size=30, weight="700", fill="#111827"),
        text(
            800,
            86,
            "Normal telemetry is learned as a short-term forecast; large future-prediction errors become anomaly evidence",
            size=15,
            fill="#455a64",
        ),
    ]

    # Data path.
    parts.append('<g filter="url(#shadow)">')
    parts.append(block(55, 160, 210, 88, "Clean telemetry", "1 Hz CSV rows", "#e3f2fd", "#0d47a1"))
    parts.append(block(330, 160, 230, 88, "Feature selection", f"D = {FEATURES_USED}, GPU excluded", "#e8f5e9", "#1b5e20"))
    parts.append(block(625, 160, 230, 88, "Window builder", f"past T={T}s, future H={H}s", "#fff3e0", "#e65100"))
    parts.append(block(920, 160, 245, 88, "Normalize", "mu/sigma from train split", "#f3e5f5", "#4a148c"))
    parts.append(block(1230, 160, 285, 88, "TCN input tensor", f"B x {FEATURES_USED} x {T}", "#eceff1", "#263238"))
    parts.append("</g>")
    for x1, x2 in [(265, 330), (560, 625), (855, 920), (1165, 1230)]:
        parts.append(line(x1, 204, x2, 204))

    # Model internals.
    parts.append(text(800, 330, "Forecaster internals", size=22, weight="700", fill="#111827"))
    parts.append('<g filter="url(#shadow)">')
    parts.append(block(75, 390, 230, 95, "Input projection", f"Conv1d 1x1: {FEATURES_USED} -> {HIDDEN_CHANNELS}", "#e0f2f1", "#004d40"))
    parts.append(block(370, 390, 170, 95, "TCN block 1", "dilation=1", "#ffffff", "#1f2937"))
    parts.append(block(570, 390, 170, 95, "TCN block 2", "dilation=2", "#ffffff", "#1f2937"))
    parts.append(block(770, 390, 170, 95, "TCN block 3", "dilation=4", "#ffffff", "#1f2937"))
    parts.append(block(970, 390, 170, 95, "TCN block 4", "dilation=8", "#ffffff", "#1f2937"))
    parts.append(block(1170, 390, 170, 95, "TCN block 5", "dilation=16", "#ffffff", "#1f2937"))
    parts.append(block(1370, 390, 170, 95, "TCN block 6", "dilation=32", "#ffffff", "#1f2937"))
    parts.append("</g>")
    for x1, x2 in [(305, 370), (540, 570), (740, 770), (940, 970), (1140, 1170), (1340, 1370)]:
        parts.append(line(x1, 438, x2, 438))

    # Block details under TCN blocks.
    detail_y = 535
    parts.append(rect(370, detail_y, 1170, 78, "#f8fafc", "#90a4ae", radius=10))
    parts.append(text(955, detail_y + 28, "Each residual block", size=17, weight="700", fill="#111827"))
    parts.append(
        text(
            955,
            detail_y + 55,
            "causal Conv1d -> ReLU -> Dropout -> causal Conv1d -> ReLU -> Dropout -> residual add",
            size=14,
            fill="#37474f",
        )
    )
    for x in [455, 655, 855, 1055, 1255, 1455]:
        parts.append(line(x, 486, x, detail_y, marker=False, dash="5 5", stroke="#90a4ae", width=1.3))

    # Forecast and scoring path.
    parts.append('<g filter="url(#shadow)">')
    parts.append(block(75, 665, 230, 90, "Last hidden state", f"B x {HIDDEN_CHANNELS}", "#ede7f6", "#311b92"))
    parts.append(block(370, 665, 230, 90, "Linear head", f"{HIDDEN_CHANNELS} -> {H} x {FEATURES_USED}", "#e1f5fe", "#01579b"))
    parts.append(block(665, 665, 230, 90, "Future forecast", f"y_hat: B x {H} x {FEATURES_USED}", "#e8f5e9", "#1b5e20"))
    parts.append(block(960, 665, 245, 90, "Feature errors", "MSE per feature, top-5 z+", "#fff8e1", "#ff6f00"))
    parts.append(block(1270, 665, 245, 90, "Event decision", "threshold + consecutive windows", "#ffebee", "#b71c1c"))
    parts.append("</g>")
    for x1, x2 in [(305, 370), (600, 665), (895, 960), (1205, 1270)]:
        parts.append(line(x1, 710, x2, 710))
    parts.append(line(1455, 486, 190, 665, stroke="#607d8b", width=1.8, dash="7 6"))

    # Side facts.
    parts.append(rect(75, 535, 230, 78, "#ffffff", "#90a4ae", radius=10))
    parts.append(text(190, 565, "Effective receptive field", size=15, weight="700", fill="#111827"))
    parts.append(text(190, 590, f"{RECEPTIVE_FIELD} time steps before final state", size=13, fill="#37474f"))

    parts.append(rect(1225, 300, 290, 58, "#ffffff", "#90a4ae", radius=10))
    parts.append(text(1370, 324, "Training objective", size=15, weight="700", fill="#111827"))
    parts.append(text(1370, 347, "minimize MSE on normal windows only", size=13, fill="#37474f"))
    parts.append(line(1305, 358, 1305, 390, marker=True, stroke="#90a4ae", width=1.5))

    parts.append(
        text(
            800,
            830,
            "Used feature groups: CPU/RAM, network throughput, disk throughput, and top-5 process CPU/RSS/threads/age. GPU fields are collected but excluded from final model features.",
            size=14,
            fill="#455a64",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_svg(), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
