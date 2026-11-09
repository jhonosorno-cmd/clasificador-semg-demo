"""Demo interactiva — Clasificador de Gestos EMG.

Trabajo de Grado · Escuela de Ingenieria Electrica y Electronica
Universidad del Valle · Dataset NinaPro DB3 (amputados transradiales).

Los datos del conjunto de prueba y las probabilidades del ensemble se leen de
`data/demo_cache.npz`, generado por `python -m src.build_demo_cache` (repositorio clasificador-semg). La demo no
necesita `data/raw/` para arrancar. La inferencia de la ventana visible si se
ejecuta en vivo sobre los 4 modelos cuando PyTorch esta disponible.

Ejecutar desde la raiz del proyecto:
    python -m streamlit run demo_app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

CACHE_PATH = ROOT / "data" / "demo_cache.npz"
CHECKPOINTS = ROOT / "checkpoints"
LIGHT_CHECKPOINTS = CHECKPOINTS / "demo"
LORO_SUMMARY = CHECKPOINTS / "loro" / "loro_cv_summary.json"
FONTS_CSS = ROOT / "assets" / "fonts.css"

# ── Nomenclatura de clases ──────────────────────────────────────────────────────
# Nombre corto para ejes y tablas; descripcion completa para tarjetas y tooltips.
GESTOS = ["Reposo", "Pulgar", "Indice", "Medio", "Anular", "Menique", "4 dedos"]
GESTOS_DESC = [
    "Reposo (sin movimiento)",
    "Flexion del pulgar",
    "Flexion del indice",
    "Flexion del dedo medio",
    "Flexion del dedo anular",
    "Flexion del menique",
    "Flexion de cuatro dedos",
]
COLORES = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]

# ── Referencias externas del trabajo (constantes documentadas, no resultados) ────
PAPER_BASELINE = 79.72   # Objetivo reportado por el paper de referencia
LATENCY_BUDGET_MS = 300  # Limite de latencia para control protesico natural

# ── Tema visual ─────────────────────────────────────────────────────────────────
UI_FONT_SANS = "'Fira Sans', 'Segoe UI', system-ui, sans-serif"
UI_FONT_MONO = "'Fira Code', 'Cascadia Mono', Consolas, ui-monospace, monospace"
PLOT_BG = "#0e1117"
PAPER_BG = "#0e1117"
UI_TEXT = "#e8eaed"
UI_MUTED = "#9aa0a6"
UI_GRID = "rgba(255,255,255,0.06)"
UI_ACCENT = "#8ab4f8"
UI_SUCCESS = "#22C55E"
UI_WARNING = "#D97706"
UI_DANGER = "#EF4444"

PLOTLY_CONFIG: dict = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3},
}


# ══════════════════════════════════════════════════════════════════════════════
# Estilos
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _font_face_css() -> str:
    """Devuelve las @font-face de Fira embebidas en base64.

    El archivo se genera una sola vez y se versiona con el proyecto, de modo que
    la demo no dependa de Google Fonts durante la sustentacion. Si no existe, se
    usan las fuentes del sistema declaradas en UI_FONT_SANS / UI_FONT_MONO.
    """
    if FONTS_CSS.exists():
        return FONTS_CSS.read_text(encoding="utf-8")
    return ""


def inject_demo_css() -> None:
    """Inyecta tipografia, fondo y estilos de tarjetas."""
    st.markdown(
        f"""
        <style>
        {_font_face_css()}

        /* El tema propio de Streamlit fija su fuente en contenedores internos, con
           mas especificidad que una regla sobre body: sin el selector descendiente
           la tipografia embebida no llega a aplicarse y todo cae a Source Sans. */
        html, body, .stApp, .stApp * {{ font-family: {UI_FONT_SANS}; }}

        /* Restituye la monoespaciada donde si corresponde. */
        code, pre, kbd, samp, .stApp code, .stApp pre {{ font-family: {UI_FONT_MONO}; }}

        .stApp {{
            background: radial-gradient(1200px 600px at 10% -10%, rgba(138,180,248,0.10), transparent 55%),
                        radial-gradient(900px 500px at 90% 0%, rgba(34,197,94,0.05), transparent 50%),
                        linear-gradient(170deg, #080b11 0%, #0e1117 48%, #060810 100%);
        }}

        .block-container {{ padding-top: 1.1rem; padding-bottom: 2.25rem; }}

        h1 {{ font-weight: 700 !important; letter-spacing: -0.02em; color: #f8f9fa !important; }}

        [data-testid="stCaption"] {{ color: {UI_MUTED} !important; }}

        [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 6px;
        }}

        [data-baseweb="tab"] {{
            border-radius: 9px; color: {UI_MUTED};
            font-weight: 600; font-size: 0.95rem;
        }}

        [aria-selected="true"] {{
            background: rgba(138,180,248,0.16) !important;
            color: #f1f3f4 !important;
        }}

        div[data-testid="stMetricValue"] {{
            font-family: {UI_FONT_MONO}; letter-spacing: -0.01em;
        }}

        .tdg-hero {{
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border-radius: 16px; padding: 18px 22px; margin-bottom: 10px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        }}

        .tdg-hero-badge {{
            display: inline-block; font-size: 0.78rem; font-weight: 700;
            letter-spacing: 0.06em; text-transform: uppercase; color: {UI_ACCENT};
            background: rgba(138,180,248,0.12); border: 1px solid rgba(138,180,248,0.22);
            border-radius: 999px; padding: 6px 12px; margin-bottom: 10px;
        }}

        .tdg-hero-title {{
            font-size: 1.85rem; line-height: 1.15; font-weight: 700;
            color: #fafafa; margin: 0 0 6px 0;
        }}

        .tdg-hero-sub {{ margin: 0; color: {UI_MUTED}; font-size: 0.95rem; }}

        .tdg-kicker {{
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: {UI_MUTED}; margin: 0 0 8px 0;
        }}

        .tdg-pred-card {{
            position: relative; overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10); border-radius: 16px;
            padding: 18px 22px; margin-bottom: 14px;
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            box-shadow: 0 14px 35px rgba(0,0,0,0.28);
        }}

        .tdg-pred-card::before {{
            content: ""; position: absolute; inset: -1px;
            background: radial-gradient(600px 200px at 10% 0%, rgba(255,255,255,0.10), transparent 55%);
            opacity: 0.65; pointer-events: none;
        }}

        .tdg-pred-label {{
            font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: {UI_MUTED}; position: relative;
        }}

        .tdg-pred-title {{
            font-size: clamp(1.6rem, 2.3vw, 2.1rem); font-weight: 800;
            letter-spacing: -0.02em; margin: 6px 0 8px 0; position: relative;
        }}

        .tdg-pred-desc {{
            font-size: 0.95rem; color: {UI_MUTED}; margin: 0 0 10px 0; position: relative;
        }}

        .tdg-pred-meta {{
            font-size: 0.92rem; color: #cfd3d7; margin-top: 4px; position: relative;
        }}

        .tdg-pred-meta strong {{ font-family: {UI_FONT_MONO}; font-weight: 600; }}

        .tdg-prov {{
            font-family: {UI_FONT_MONO}; font-size: 0.80rem; color: {UI_MUTED};
            border-left: 3px solid rgba(138,180,248,0.45); padding: 6px 0 6px 12px;
            margin-bottom: 16px;
        }}

        .tdg-step {{
            border: 1px solid rgba(255,255,255,0.10); border-radius: 12px;
            background: rgba(255,255,255,0.03); padding: 12px 14px; height: 100%;
        }}

        .tdg-step-n {{
            font-family: {UI_FONT_MONO}; font-size: 0.75rem; font-weight: 600;
            color: {UI_ACCENT};
        }}

        .tdg-step-t {{ font-weight: 600; font-size: 0.95rem; margin: 4px 0 2px 0; }}
        .tdg-step-d {{ font-size: 0.82rem; color: {UI_MUTED}; line-height: 1.35; }}

        .js-plotly-plot .plotly .modebar {{ top: 6px !important; right: 6px !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_style(
    fig: go.Figure,
    *,
    height: int | None = None,
    margin: dict | None = None,
    with_polars: bool = False,
    cartesian_axes: bool = True,
) -> go.Figure:
    """Aplica tipografia y fondo coherentes a una figura Plotly.

    Args:
        fig: Figura a estilizar.
        height: Alto en pixeles; None conserva el del layout.
        margin: Margenes; None usa un valor compacto por defecto.
        with_polars: True para figuras de coordenadas polares (radar).
        cartesian_axes: False evita tocar los ejes (figuras sin ejes reales).

    Returns:
        La misma figura, ya estilizada.
    """
    fig.update_layout(
        font=dict(family=UI_FONT_SANS, color=UI_TEXT, size=12),
        title_font=dict(size=14, color="#fafafa", family=UI_FONT_SANS),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        margin=margin or dict(l=12, r=12, t=40, b=12),
        hoverlabel=dict(bgcolor="#161a23", font_family=UI_FONT_SANS),
    )
    if height is not None:
        fig.update_layout(height=height)
    if with_polars:
        fig.update_layout(polar=dict(bgcolor=PAPER_BG))
        fig.update_polars(
            radialaxis=dict(gridcolor=UI_GRID, linecolor=UI_GRID),
            angularaxis=dict(gridcolor=UI_GRID, linecolor=UI_GRID),
        )
        return fig
    if not cartesian_axes:
        return fig
    axis_style = dict(
        showgrid=True, gridcolor=UI_GRID, zeroline=False, linecolor=UI_GRID,
        tickfont=dict(color=UI_MUTED, size=11), title_font=dict(color=UI_MUTED, size=12),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def empty_fig(title: str, message: str, height: int = 290) -> go.Figure:
    """Figura de marcador de posicion con un mensaje centrado."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[dict(
            text=message, showarrow=False, x=0.5, y=0.55, xref="paper", yref="paper",
            font=dict(color=UI_MUTED, size=13, family=UI_FONT_SANS),
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return apply_plotly_style(fig, height=height, cartesian_axes=False)


# ══════════════════════════════════════════════════════════════════════════════
# Carga de recursos
# ══════════════════════════════════════════════════════════════════════════════

CACHE_VERSION = "2024-demo-4m-v1"


@st.cache_data(show_spinner="Cargando conjunto de prueba precomputado...")
def load_cache(version: str) -> dict:
    """Carga `data/demo_cache.npz` en memoria.

    Args:
        version: Etiqueta de version del cache; participa en la clave de la
            cache de Streamlit para invalidarla si cambia el contenido.

    Returns:
        Diccionario con los arrays del cache.

    Raises:
        FileNotFoundError: Si el cache no ha sido generado.
    """
    del version  # solo interviene en la clave de cache
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CACHE_PATH}. Generalo con:  python -m src.build_demo_cache (repositorio clasificador-semg)"
        )
    with np.load(CACHE_PATH) as data:
        return {k: data[k] for k in data.files}


@st.cache_resource(show_spinner="Cargando los 4 modelos del ensemble...")
def load_ensemble() -> tuple | None:
    """Carga los 4 modelos del ensemble para inferencia en vivo.

    Prefiere los checkpoints ligeros de `checkpoints/demo/` (solo pesos) y cae
    a los originales si no existen. Devuelve None si PyTorch no esta instalado,
    en cuyo caso la demo funciona con las probabilidades del cache.

    Returns:
        Tupla (torch, lista de (alias, modelo, dim_features)) o None.
    """
    try:
        import torch

        from src.models_cyborg import CyborgHybrid
        from src.models_v18 import CyborgHybridAttention
    except Exception:
        return None

    spec = [
        ("v10", CyborgHybrid, "best_model_cyborg.pth", 48),
        ("v15", CyborgHybrid, "best_model_v15_augmentation.pth", 48),
        ("v17", CyborgHybrid, "best_model_v17_features.pth", 132),
        ("v18", CyborgHybridAttention, "best_model_v18_attention.pth", 48),
    ]
    models = []
    for alias, model_cls, ckpt_name, feat_dim in spec:
        path = LIGHT_CHECKPOINTS / ckpt_name
        if not path.exists():
            path = CHECKPOINTS / ckpt_name
        if not path.exists():
            return None
        model = model_cls(manual_features_dim=feat_dim)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append((alias, model, feat_dim))

    # Warm-up: las primeras pasadas incluyen inicializacion perezosa de kernels y
    # falsearian la latencia que se muestra en pantalla.
    with torch.no_grad():
        dummy_sig = torch.zeros(1, 1, 125, 13)
        for _ in range(15):
            for _, model, feat_dim in models:
                model(dummy_sig, torch.zeros(1, feat_dim))
    return torch, models


@st.cache_data(show_spinner=False)
def load_loro() -> dict | None:
    """Carga el resumen de validacion cruzada leave-one-subject-out."""
    if not LORO_SUMMARY.exists():
        return None
    return json.loads(LORO_SUMMARY.read_text(encoding="utf-8"))


def infer_live(
    ensemble: tuple, signals: np.ndarray, feat48: np.ndarray, feat132: np.ndarray, idx: int
) -> tuple[np.ndarray, float]:
    """Ejecuta el ensemble sobre una unica ventana y cronometra la pasada.

    Args:
        ensemble: Tupla devuelta por `load_ensemble`.
        signals: (N, 1, 125, 13) ventanas EMG.
        feat48: (N, 48) features manuales.
        feat132: (N, 132) features extendidas.
        idx: Indice de la ventana a evaluar.

    Returns:
        Tupla (probabilidades (7,), latencia en milisegundos).
    """
    torch, models = ensemble
    sig_t = torch.from_numpy(signals[idx : idx + 1])
    f48_t = torch.from_numpy(feat48[idx : idx + 1])
    f132_t = torch.from_numpy(feat132[idx : idx + 1])

    t0 = time.perf_counter()
    with torch.no_grad():
        total = None
        for _, model, feat_dim in models:
            feats = f132_t if feat_dim == 132 else f48_t
            p = torch.softmax(model(sig_t, feats), dim=1)
            total = p if total is None else total + p
        probs = (total / len(models)).numpy()[0]
    return probs, (time.perf_counter() - t0) * 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Regla de decision
# ══════════════════════════════════════════════════════════════════════════════

def predict(probs: np.ndarray, theta: float | None) -> np.ndarray:
    """Convierte probabilidades en predicciones.

    Args:
        probs: (N, 7) probabilidades softmax del ensemble.
        theta: None para argmax estandar; en caso contrario, umbral minimo de
            confianza que debe superar la clase de reposo para ser elegida.

    Returns:
        (N,) predicciones enteras 0-6.
    """
    probs_2d = np.atleast_2d(probs)
    if theta is None:
        preds = np.argmax(probs_2d, axis=1)
    else:
        active = np.argmax(probs_2d[:, 1:], axis=1) + 1
        preds = np.where(probs_2d[:, 0] > theta, 0, active)
    return preds.astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# Figuras
# ══════════════════════════════════════════════════════════════════════════════

def fig_emg_window(signal_125x13: np.ndarray, channels: list[int] | None = None) -> go.Figure:
    """Traza los canales EMG seleccionados de la ventana actual.

    Args:
        signal_125x13: (125, 13) ventana EMG mas canal RMS.
        channels: Indices 0-based a mostrar; None muestra los 12 canales EMG.
    """
    t_ms = np.arange(125) * 2  # 500 Hz -> 2 ms por muestra
    channels = channels if channels else list(range(12))

    fig = go.Figure()
    for ch in channels:
        fig.add_trace(go.Scatter(
            x=t_ms, y=signal_125x13[:, ch], mode="lines",
            line=dict(width=0.9), name=f"Ch{ch+1}",
        ))
    fig.update_layout(
        title=f"Senal EMG — ventana actual (250 ms) · {len(channels)} canales",
        xaxis_title="Tiempo (ms)",
        yaxis_title="Amplitud (z-score)",
        legend=dict(orientation="h", y=-0.3, font_size=9, font_color=UI_MUTED),
    )
    return apply_plotly_style(fig, height=270, margin=dict(l=10, r=10, t=38, b=38))


def fig_cyborg_features(feat_48: np.ndarray) -> go.Figure:
    """Radar de las features manuales (RMS, ZC, WL) por canal.

    El vector 48D se organiza por canal como [RMS, ZC, SSC, WL]. Cada feature se
    normaliza a [0, 1] entre canales para que los ejes sean comparables. SSC se
    omite por claridad: es fuertemente redundante con ZC.

    Args:
        feat_48: (48,) features manuales de la ventana actual.
    """
    labels = [f"Ch{i+1}" for i in range(12)]

    def _norm(v: np.ndarray) -> np.ndarray:
        return (v - v.min()) / (v.max() - v.min() + 1e-8)

    series = [
        (_norm(feat_48[0::4]), "RMS", "#4CAF50", "rgba(76,175,80,0.15)"),
        (_norm(feat_48[1::4]), "ZC", "#FF9800", "rgba(255,152,0,0.15)"),
        (_norm(feat_48[3::4]), "WL", "#2196F3", "rgba(33,150,243,0.15)"),
    ]

    fig = go.Figure()
    for values, name, color, fill in series:
        fig.add_trace(go.Scatterpolar(
            r=np.append(values, values[0]),
            theta=labels + [labels[0]],
            fill="toself", fillcolor=fill,
            line=dict(color=color, width=1.8), name=name,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont_size=7),
            angularaxis=dict(tickfont_size=9),
            bgcolor=PAPER_BG,
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font_size=9, font_color=UI_MUTED),
        title=dict(text="Features manuales por canal (normalizadas)", font_size=12),
    )
    return apply_plotly_style(fig, height=280, margin=dict(l=20, r=20, t=48, b=52), with_polars=True)


def fig_confidence_bars(probs: np.ndarray, top_n: int | None = None) -> go.Figure:
    """Barras horizontales de confianza por clase.

    Args:
        probs: (7,) probabilidades del ensemble para la ventana actual.
        top_n: None grafica las 7 clases en orden fijo; un entero grafica solo
            las `top_n` mas probables, ordenadas de mayor a menor.
    """
    if top_n is None:
        order = np.arange(7)[::-1]  # Reposo arriba en un eje horizontal
        title = "Confianza por clase (7 clases)"
    else:
        order = np.argsort(probs)[-top_n:]
        title = f"Top-{top_n} confianza por clase"

    fig = go.Figure(go.Bar(
        x=probs[order],
        y=[GESTOS[i] for i in order],
        orientation="h",
        marker_color=[COLORES[i] for i in order],
        text=[f"{probs[i]*100:.1f}%" for i in order],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis=dict(range=[0, 1.18], tickformat=".0%"))
    return apply_plotly_style(fig, height=270, margin=dict(l=10, r=10, t=38, b=12))


def fig_prediction_timeline(history: list[dict]) -> go.Figure:
    """Serie temporal de las predicciones recientes, marcando aciertos y errores."""
    if not history:
        return empty_fig("Timeline de predicciones recientes",
                         "Navega o reproduce para ver la serie temporal")

    fig = go.Figure(go.Scatter(
        x=[h["step"] for h in history],
        y=[h["pred"] for h in history],
        mode="markers+lines",
        marker=dict(
            size=9,
            color=[COLORES[h["pred"]] for h in history],
            symbol=["circle" if h["correct"] else "x" for h in history],
            line=dict(width=1, color="white"),
        ),
        line=dict(color="#9aa0a6", width=1),
        text=[
            f"Ventana #{h['idx']+1}<br>Prediccion: {GESTOS[h['pred']]}<br>"
            f"Real: {GESTOS[h['real']]}<br>Confianza: {h['conf']*100:.1f}%"
            for h in history
        ],
        hoverinfo="text",
        name="Prediccion",
    ))
    fig.update_layout(
        title="Timeline de predicciones recientes",
        xaxis_title="Paso de simulacion",
        yaxis=dict(title="Clase predicha", tickmode="array",
                   tickvals=list(range(7)), ticktext=GESTOS),
    )
    return apply_plotly_style(fig, height=290)


def fig_confidence_trend(history: list[dict]) -> go.Figure:
    """Evolucion de la confianza Top-1 en las ventanas recorridas."""
    if not history:
        return empty_fig("Tendencia de confianza (Top-1)",
                         "Aun no hay puntos — avanza ventanas o activa reproduccion")

    fig = go.Figure(go.Scatter(
        x=[h["step"] for h in history],
        y=[h["conf"] * 100 for h in history],
        mode="lines+markers",
        line=dict(color="#00BCD4", width=2),
        marker=dict(size=5, color="#00BCD4"),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.18)",
        name="Confianza",
        hovertemplate="Paso %{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Tendencia de confianza (Top-1)",
        xaxis_title="Paso de simulacion",
        yaxis=dict(title="Confianza (%)", range=[0, 100]),
    )
    return apply_plotly_style(fig, height=290)


def fig_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Matriz de confusion normalizada por fila (recall por clase en la diagonal)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(7))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float),
                       where=row_sums > 0) * 100

    fig = px.imshow(
        cm_pct,
        labels=dict(x="Predicho", y="Real", color="%"),
        x=GESTOS, y=GESTOS,
        color_continuous_scale=[
            [0.00, PLOT_BG], [0.35, "#1e3a5c"], [0.65, "#2563eb"], [1.00, UI_ACCENT],
        ],
        text_auto=".1f", zmin=0, zmax=100,
    )
    fig.update_layout(title=f"Matriz de confusion — Accuracy {np.mean(y_true == y_pred)*100:.2f}%")
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=UI_MUTED)))
    return apply_plotly_style(fig, height=420, margin=dict(l=10, r=10, t=48, b=12))


def fig_theta_curve(grid: np.ndarray, val_acc: np.ndarray, val_bal: np.ndarray,
                    theta_opt: float) -> go.Figure:
    """Curva de accuracy frente al umbral de reposo, medida en validacion."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=val_acc * 100, mode="lines",
                             line=dict(color=UI_ACCENT, width=2), name="Accuracy (val)"))
    fig.add_trace(go.Scatter(x=grid, y=val_bal * 100, mode="lines",
                             line=dict(color="#22C55E", width=2, dash="dot"),
                             name="Balanced acc. (val)"))
    fig.add_vline(x=theta_opt, line_color="#F9A825", line_dash="dash",
                  annotation_text=f"theta* = {theta_opt:.2f}", annotation_position="top right")
    fig.update_layout(
        title="Seleccion del umbral de reposo — barrido sobre validacion (rep 5)",
        xaxis_title="Umbral theta sobre P(reposo)",
        yaxis_title="Metrica (%)",
        legend=dict(orientation="h", y=-0.25, font_color=UI_MUTED),
    )
    return apply_plotly_style(fig, height=330, margin=dict(l=12, r=12, t=45, b=55))


def fig_loro(loro: dict) -> go.Figure:
    """Accuracy leave-one-subject-out por modelo, con desviacion entre folds."""
    alias = {"v10": "v10 CyborgHybrid", "v17": "v17 Features 132D",
             "v18": "v18 Self-Attention", "ensemble_4m": "Ensemble 4M"}
    keys = [k for k in ["v10", "v17", "v18", "ensemble_4m"] if k in loro]
    means = [loro[k]["accuracy"]["mean"] * 100 for k in keys]
    stds = [loro[k]["accuracy"]["std"] * 100 for k in keys]
    colors = ["#1565C0"] * len(keys)
    if "ensemble_4m" in keys:
        colors[keys.index("ensemble_4m")] = "#F9A825"

    fig = go.Figure(go.Bar(
        x=[alias[k] for k in keys], y=means,
        error_y=dict(type="data", array=stds, color=UI_MUTED, thickness=1.5, width=8),
        marker_color=colors,
        text=[f"{m:.2f}%" for m in means],
        textposition="outside",
        customdata=stds,
        hovertemplate="%{x}<br>%{y:.2f}% ± %{customdata:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Validacion cruzada leave-one-subject-out (6 folds)",
        yaxis=dict(title="Accuracy (%)", range=[60, 78]),
    )
    return apply_plotly_style(fig, height=340, margin=dict(l=12, r=12, t=45, b=12))


def fig_loro_folds(loro: dict) -> go.Figure:
    """Accuracy por fold de cada modelo, para exponer la dispersion inter-sujeto."""
    alias = {"v10": "v10", "v17": "v17", "v18": "v18", "ensemble_4m": "Ensemble 4M"}
    colors = {"v10": "#2196F3", "v17": "#FF9800", "v18": "#9C27B0", "ensemble_4m": "#F9A825"}

    fig = go.Figure()
    for key in [k for k in ["v10", "v17", "v18", "ensemble_4m"] if k in loro]:
        values = np.array(loro[key]["accuracy"]["values"]) * 100
        fig.add_trace(go.Scatter(
            x=[f"Fold {i+1}" for i in range(len(values))], y=values,
            mode="lines+markers", name=alias[key],
            line=dict(color=colors[key], width=2 if key == "ensemble_4m" else 1.4),
            marker=dict(size=8 if key == "ensemble_4m" else 6),
        ))
    fig.update_layout(
        title="Accuracy por fold — cada fold deja un sujeto fuera del entrenamiento",
        yaxis=dict(title="Accuracy (%)"),
        legend=dict(orientation="h", y=-0.2, font_color=UI_MUTED),
    )
    return apply_plotly_style(fig, height=340, margin=dict(l=12, r=12, t=45, b=50))


VERSION_HISTORY = [
    ("v7 CNN+BiLSTM", 69.95, False),
    ("v10 CyborgHybrid", 71.32, False),
    ("v11 Focal agresivo", 51.99, True),
    ("v12 Focal moderado", 70.17, False),
    ("v13 Multi-sujeto", 68.91, True),
    ("v14 Fine-tuning", 70.69, False),
    ("v15 Augmentation", 70.74, False),
    ("Ensemble 2M", 72.41, False),
    ("v17 Features 132D", 71.89, False),
    ("Ensemble 3M pond.", 72.57, False),
    ("v18 Self-Attention", 71.63, False),
    ("Ensemble 4M", 72.78, False),
    ("Ensemble 4M + umbral", 72.88, False),
    ("v19 TCN", 69.80, True),
    ("v20 Spectral 108D", 70.11, True),
    ("v22 Cascada", 70.27, True),
    ("v23 Transformer", 71.89, False),
    ("v24 Conformer", 71.00, True),
    ("v25 STFT+CNN2D", 68.70, True),
    ("v26 LS+SWA", 71.47, False),
]


def fig_evolution(best_label: str, best_value: float, live: dict[str, float]) -> go.Figure:
    """Barras con la evolucion de las versiones evaluadas y los hitos anotados.

    Args:
        best_label: Nombre de la configuracion que marca el maximo.
        best_value: Accuracy del maximo, para la linea de referencia.
        live: Valores recalculados en esta ejecucion para las entradas del
            ensemble, de modo que la grafica no pueda contradecir a la pestana
            de metricas si el cache se regenera.
    """
    nombres = [d[0] for d in VERSION_HISTORY]
    valores = [live.get(d[0], d[1]) for d in VERSION_HISTORY]
    cols = ["#EF9A9A" if d[2] else "#1565C0" for d in VERSION_HISTORY]
    if best_label in nombres:
        cols[nombres.index(best_label)] = "#F9A825"

    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h", marker_color=cols,
        text=[f"{v:.2f}%" for v in valores], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))

    fig.add_vline(x=PAPER_BASELINE, line_color="red", line_dash="dash",
                  annotation_text=f"Objetivo {PAPER_BASELINE:.2f}%",
                  annotation_position="top right")
    fig.add_vline(x=best_value, line_color="#F9A825", line_dash="dash",
                  annotation_text=f"Maximo {best_value:.2f}%",
                  annotation_position="bottom right")

    hitos = [
        (51.99, "v11 Focal agresivo", "Hito 1<br>Caida Focal/Z-Score", "#FF5722", -70, 0),
        (71.32, "v10 CyborgHybrid", "Hito 2<br>Arq. Cyborg", "#4CAF50", 65, -28),
        (68.91, "v13 Multi-sujeto", "Hito 3<br>Multi-sujeto", "#9C27B0", 65, -28),
    ]
    for x, y, text, color, ax, ay in hitos:
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowcolor=color, ax=ax, ay=ay, font=dict(size=10, color=color),
            bgcolor="rgba(0,0,0,0.35)", bordercolor=color, borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        title=f"Evolucion de resultados — {len(VERSION_HISTORY)} configuraciones evaluadas",
        xaxis=dict(range=[48, 88], title="Test Accuracy (%) — S1, repeticion 6"),
    )
    fig.update_yaxes(tickfont=dict(color=UI_MUTED))
    return apply_plotly_style(fig, height=620, margin=dict(l=12, r=112, t=48, b=12))


# ══════════════════════════════════════════════════════════════════════════════
# Analisis de errores
# ══════════════════════════════════════════════════════════════════════════════

def get_top_confusions(y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 3) -> list[dict]:
    """Devuelve las confusiones mas frecuentes fuera de la diagonal.

    El orden es por numero absoluto de casos, no por porcentaje: una clase con
    dos muestras y un error alcanzaria el 50% sin ser un error relevante.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones del modelo.
        top_n: Numero de confusiones a devolver.

    Returns:
        Lista de dicts con las claves real, pred, count, total y pct.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(7))
    confusions = [
        {"real": i, "pred": j, "count": int(cm[i, j]),
         "total": int(cm[i].sum()), "pct": cm[i, j] / cm[i].sum() * 100}
        for i in range(7) for j in range(7)
        if i != j and cm[i, j] > 0 and cm[i].sum() > 0
    ]
    confusions.sort(key=lambda c: c["count"], reverse=True)
    return confusions[:top_n]


def error_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Caracteriza la distribucion global de errores del clasificador.

    Distingue los tres modos de fallo posibles: un gesto que se pierde como
    reposo, un reposo que se interpreta como gesto, y una confusion entre dos
    gestos distintos.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones del modelo.

    Returns:
        Dict con el total de errores y el reparto porcentual por modo.
    """
    errores = y_true != y_pred
    total = int(errores.sum())
    if total == 0:
        return {"total": 0, "gesto_a_reposo": 0.0, "reposo_a_gesto": 0.0, "entre_gestos": 0.0}
    gesto_a_reposo = int(((y_true != 0) & (y_pred == 0)).sum())
    reposo_a_gesto = int(((y_true == 0) & (y_pred != 0)).sum())
    entre_gestos = total - gesto_a_reposo - reposo_a_gesto
    return {
        "total": total,
        "gesto_a_reposo": gesto_a_reposo / total * 100,
        "reposo_a_gesto": reposo_a_gesto / total * 100,
        "entre_gestos": entre_gestos / total * 100,
    }


def class_mean_rms(feat48: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Amplitud media por clase, promediando el RMS de los 12 canales.

    En el vector 48D el RMS de cada canal ocupa las posiciones 0, 4, 8, ...
    Esta magnitud explica por si sola los errores contra la clase de reposo:
    un gesto cuya envolvente apenas supera la del reposo es indistinguible
    para el clasificador.

    Args:
        feat48: (N, 48) features manuales.
        labels: (N,) etiquetas reales.

    Returns:
        (7,) RMS medio por clase.
    """
    rms = feat48[:, 0::4].mean(axis=1)
    return np.array([rms[labels == c].mean() if np.any(labels == c) else 0.0
                     for c in range(7)])


def confusion_evidence(err: dict, corr: np.ndarray, rms: np.ndarray) -> str:
    """Evidencia cuantitativa, medida sobre el conjunto de prueba, de una confusion.

    Para los pares que involucran reposo la magnitud relevante es la amplitud:
    se compara el RMS medio de ambas clases. Para las confusiones entre dos
    gestos lo relevante es la forma del patron, y se reporta la correlacion
    entre sus perfiles medios de features.

    Args:
        err: Entrada devuelta por `get_top_confusions`.
        corr: (7, 7) correlacion entre perfiles medios de clase.
        rms: (7,) RMS medio por clase.

    Returns:
        Texto con la evidencia lista para mostrar.
    """
    real, pred = err["real"], err["pred"]
    if real == 0 or pred == 0:
        gesto = pred if real == 0 else real
        ratio = rms[gesto] / (rms[0] + 1e-8)
        return (f"amplitud media del gesto {ratio:.2f}× la del reposo "
                f"(RMS {rms[gesto]:.3f} frente a {rms[0]:.3f})")
    return f"correlacion entre perfiles medios de features r = {corr[real, pred]:+.2f}"


# Mecanismo anatomico por par (clase_real, clase_predicha). Descripciones
# cualitativas: las cifras que acompanan a cada par en la interfaz se calculan
# sobre el propio conjunto de prueba.
_ANAT: dict[tuple[int, int], str] = {
    (0, 1): "El abductor pollicis longus (APL) y el extensor pollicis brevis (EPB) presentan "
            "actividad tonica residual en reposo en amputados transradiales. Esa actividad "
            "supera el umbral de ruido y solapa el patron de inicio del pulgar.",
    (0, 2): "La actividad tonica del flexor digitorum profundus (FDP) es habitual en el miembro "
            "residual en reposo. Reproduce en baja amplitud los patrones de inicio del indice.",
    (0, 3): "Co-activacion del flexor digitorum superficialis (FDS) durante el reposo activo: el "
            "munon mantiene una tension basal que recluta las mismas fibras del dedo medio.",
    (1, 2): "El oponente del pulgar y el primer interoseo dorsal comparten inervacion del nervio "
            "mediano. Los electrodos de superficie capturan senal mezclada de ambos durante la "
            "flexion aislada del pulgar.",
    (1, 6): "La flexion de cuatro dedos implica co-activacion del flexor pollicis longus (FPL), "
            "motor principal del pulgar, de modo que ambos gestos comparten generadores musculares.",
    (2, 3): "Crosstalk FDP/FDS: los tendones del indice y del dedo medio comparten vaina sinovial, "
            "y los electrodos de superficie recogen senal mezclada al contraer cualquiera de ellos.",
    (1, 0): "La flexion del pulgar se apoya en el flexor pollicis longus, un musculo profundo que "
            "los electrodos de superficie captan de forma atenuada. En el miembro residual, ademas, "
            "el reclutamiento voluntario es menor, de modo que la envolvente del gesto se acerca a "
            "la del reposo y el clasificador opta por la clase mayoritaria.",
    (2, 0): "Baja amplitud EMG del indice en amputados: la deaferenciacion sensorial reduce el "
            "reclutamiento de unidades motoras y aproxima la senal al nivel de reposo.",
    (3, 0): "La flexion aislada del dedo medio depende de fasciculos del FDP situados en un plano "
            "profundo del antebrazo. Sin retroalimentacion propioceptiva el sujeto amputado genera "
            "contracciones debiles que el modelo asimila a reposo.",
    (4, 0): "El anular carece de un flexor independiente: se mueve por fasciculos compartidos del "
            "FDP y por lumbricales de pequena seccion, de modo que su contraccion aislada exige un "
            "esfuerzo que el miembro residual rara vez sostiene durante la ventana completa.",
    (6, 0): "En la flexion de cuatro dedos la fuerza se reparte entre varios fasciculos, de modo "
            "que ningun canal individual alcanza el pico que si produce un gesto concentrado. "
            "Cuando la contraccion es suave el patron queda por debajo del umbral de deteccion.",
    (3, 2): "Inervacion compartida: las ramas del nervio mediano para el dedo medio y el anular son "
            "anatomicamente adyacentes, y su activacion difiere poco en amplitud.",
    (3, 4): "Los lumbricales del dedo medio y del anular reciben ramas muy proximas del nervio "
            "mediano, por lo que sus patrones de activacion son casi paralelos.",
    (4, 3): "Contraparte simetrica del par anterior. La bidireccionalidad indica que el solapamiento "
            "es mutuo: ninguno de los dos gestos resulta mas facil de aislar que el otro.",
    (4, 5): "Distribucion ulnar: el nervio ulnar inerva conjuntamente la musculatura hipotenar del "
            "menique y parte de los interoseos del anular, lo que produce co-activacion sinergica.",
    (5, 4): "Contraparte del par anterior. Los canales de la zona hipotenar no separan con fiabilidad "
            "ambos gestos en contracciones de baja fuerza.",
    (5, 0): "El menique es el digito con menor area de representacion en el homunculo motor. En "
            "amputados transradiales su senal residual puede quedar por debajo del umbral util.",
    (6, 1): "La flexion de cuatro dedos recluta APL y FPL, los mismos motores del pulgar; cuando la "
            "actividad tenar domina los canales anteriores el gesto colectivo colapsa en pulgar.",
    (6, 3): "Durante la flexion de cuatro dedos el dedo medio aporta la mayor fuerza (mayor seccion "
            "transversal del FDP), y su pico de amplitud puede dominar sobre el patron colectivo.",
}

# Respaldos por direccion de la confusion: el mecanismo dominante no es el mismo
# cuando un gesto se pierde como reposo que cuando dos gestos se solapan entre si.
_ANAT_GESTO_A_REPOSO = (
    "La senal residual del gesto no alcanza a separarse del nivel basal: en amputados "
    "transradiales la ausencia de retroalimentacion propioceptiva reduce el reclutamiento de "
    "unidades motoras, y el clasificador resuelve la ambiguedad a favor de la clase mayoritaria."
)
_ANAT_REPOSO_A_GESTO = (
    "Actividad tonica residual del miembro: el munon no llega a relajarse por completo y mantiene "
    "una contraccion basal que reproduce el inicio de un gesto voluntario."
)
_ANAT_ENTRE_GESTOS = (
    "Probable crosstalk entre los flexores superficiales (FDS) y profundos (FDP): los electrodos de "
    "superficie integran la actividad de musculos adyacentes durante contracciones aisladas. En "
    "amputados transradiales el efecto se acentua por la reorganizacion cortical post-amputacion."
)


def anat_text(real: int, pred: int) -> str:
    """Mecanismo anatomico documentado para un par de confusion.

    Args:
        real: Clase verdadera.
        pred: Clase predicha por el modelo.

    Returns:
        Descripcion del mecanismo; si el par concreto no esta documentado, el
        respaldo correspondiente a la direccion de la confusion.
    """
    if (real, pred) in _ANAT:
        return _ANAT[(real, pred)]
    if pred == 0:
        return _ANAT_GESTO_A_REPOSO
    if real == 0:
        return _ANAT_REPOSO_A_GESTO
    return _ANAT_ENTRE_GESTOS


# ══════════════════════════════════════════════════════════════════════════════
# Paneles
# ══════════════════════════════════════════════════════════════════════════════

def render_controls(cache: dict, ensemble: tuple | None) -> float | None:
    """Dibuja la configuracion global del ensemble en el cuerpo de la pagina.

    Deliberadamente no usa la barra lateral: Streamlit la colapsa por debajo de
    unos 1300 px de ancho y en ese estado no deja ni el control para abrirla. En
    un proyector a 1280x720 el selector de la regla de decision — el que produce
    el resultado principal del trabajo — quedaria inaccesible.

    Args:
        cache: Contenido de `demo_cache.npz`.
        ensemble: Modelos cargados, o None si se opera solo con el cache.

    Returns:
        El umbral activo, o None si la regla es argmax.
    """
    theta_opt = float(cache["theta_opt"])

    with st.container(border=True):
        col_regla, col_theta, col_estado = st.columns([2.2, 1.6, 2.2])

        with col_regla:
            modo = st.radio(
                "Regla de decision",
                ["Argmax (4M uniforme)", f"Umbral de reposo (theta* = {theta_opt:.2f})"],
                horizontal=False,
                help=(
                    "Argmax elige simplemente la clase mas probable. La regla con umbral "
                    "predice reposo siempre que P(reposo) supere theta — aunque otra clase "
                    "tenga mas probabilidad — y en caso contrario elige el gesto activo mas "
                    "probable. theta se optimizo sobre el conjunto de validacion "
                    "(repeticion 5), nunca sobre el de prueba."
                ),
            )
        usar_umbral = modo.startswith("Umbral")

        with col_theta:
            theta = theta_opt
            if usar_umbral:
                theta = st.slider(
                    "Umbral theta", min_value=0.10, max_value=0.95,
                    value=theta_opt, step=0.02,
                    help="Ajustar este valor con el conjunto de prueba a la vista seria "
                         "sobreajuste; el valor por defecto es el optimo hallado en validacion.",
                )
                if abs(theta - theta_opt) > 1e-6:
                    st.caption("⚠ Valor manual, fuera del optimo de validacion.")
            else:
                st.caption("La regla argmax no usa umbral.")

        with col_estado:
            if ensemble is not None:
                st.markdown(
                    f'<div style="background:{UI_SUCCESS}1a;border:1px solid {UI_SUCCESS}55;'
                    f'border-radius:8px;padding:8px 12px;font-size:0.86rem;font-weight:600;'
                    f'color:{UI_SUCCESS};">⚡ PyTorch activo — la ventana visible se infiere '
                    f'en vivo sobre CPU</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:{UI_WARNING}1a;border:1px solid {UI_WARNING}55;'
                    f'border-radius:8px;padding:8px 12px;font-size:0.86rem;font-weight:600;'
                    f'color:{UI_WARNING};">💾 PyTorch no disponible — se usan las '
                    f'probabilidades precomputadas</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Latencia medida: p50 {float(cache['latency_p50']):.1f} ms · "
                f"p95 {float(cache['latency_p95']):.1f} ms · "
                f"{int(cache['latency_runs'])} repeticiones tras warm-up. "
                f"Validacion de theta: {len(cache['val_labels']):,} ventanas (rep 5)."
            )

    return theta if usar_umbral else None


@st.fragment
def render_simulation(cache: dict, ensemble: tuple | None, theta: float | None) -> None:
    """Panel de simulacion ventana a ventana.

    Se ejecuta como fragmento para que la reproduccion automatica refresque solo
    esta seccion en lugar de re-renderizar la aplicacion completa.

    Args:
        cache: Contenido de `demo_cache.npz`.
        ensemble: Modelos cargados, o None.
        theta: Umbral de reposo activo, o None para argmax.
    """
    signals = cache["signals"]
    feat48 = cache["feat48"]
    feat132 = cache["feat132"]
    y_true = cache["labels"]
    probs_all = cache["probs_mean"]
    n_windows = len(y_true)

    defaults = {
        "idx": 0, "play": False, "play_speed": "1x",
        "timeline_history": [], "timeline_step": 0, "last_timeline_idx": -1,
        "class_cursor": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    y_pred_all = predict(probs_all, theta)
    col_ctrl, col_info = st.columns([1, 3])

    with col_ctrl:
        st.markdown('<p class="tdg-kicker">Navegacion</p>', unsafe_allow_html=True)
        idx = st.session_state.idx

        clase_filtro = st.selectbox(
            "Ir a un gesto:", range(7),
            format_func=lambda g: f"{GESTOS[g]} — {GESTOS_DESC[g]}",
            key="clase_filtro",
        )
        indices_clase = np.where(y_true == clase_filtro)[0]
        if st.session_state.get("last_class_filtro") != clase_filtro:
            st.session_state.class_cursor = 0
            st.session_state.last_class_filtro = clase_filtro
        if st.button(f"Siguiente ejemplo de {GESTOS[clase_filtro]}",
                     width="stretch", disabled=len(indices_clase) == 0):
            # Cicla entre todas las ocurrencias en lugar de saltar siempre a la primera.
            cursor = st.session_state.class_cursor % len(indices_clase)
            st.session_state.idx = int(indices_clase[cursor])
            st.session_state.class_cursor = cursor + 1
            st.session_state.play = False
            st.rerun(scope="fragment")
        st.caption(f"{len(indices_clase)} ventanas de esta clase en el conjunto de prueba.")

        st.divider()
        c1, c2, c3 = st.columns(3)
        if c1.button("◀◀", width="stretch", help="Ir al inicio"):
            st.session_state.update(play=False, idx=0)
            st.rerun(scope="fragment")
        if c2.button("◀", width="stretch", help="Ventana anterior"):
            st.session_state.update(play=False, idx=max(0, idx - 1))
            st.rerun(scope="fragment")
        if c3.button("▶", width="stretch", help="Ventana siguiente"):
            st.session_state.update(play=False, idx=min(n_windows - 1, idx + 1))
            st.rerun(scope="fragment")

        col_r1, col_r2 = st.columns(2)
        if col_r1.button("🎲 Aleatoria", width="stretch"):
            st.session_state.update(play=False, idx=int(np.random.randint(0, n_windows)))
            st.rerun(scope="fragment")
        if col_r2.button("⚠ Peor error", width="stretch",
                         help="Ventana mal clasificada con mayor confianza: el caso mas "
                              "desfavorable para el modelo."):
            errors = np.where(y_pred_all != y_true)[0]
            if len(errors):
                worst = errors[np.argmax(probs_all[errors, y_pred_all[errors]])]
                st.session_state.update(play=False, idx=int(worst))
            st.rerun(scope="fragment")

        st.divider()
        col_play_a, col_play_b = st.columns([2, 1])
        with col_play_a:
            if st.button("⏸ Pausar" if st.session_state.play else "▶ Reproducir",
                         width="stretch", type="primary"):
                st.session_state.play = not st.session_state.play
                st.rerun(scope="fragment")
        with col_play_b:
            st.selectbox("Vel.", options=["0.5x", "1x", "2x"], key="play_speed",
                         label_visibility="collapsed")

        st.divider()
        idx = st.session_state.idx
        st.metric("Ventana", f"{idx + 1} / {n_windows}")
        st.metric("Clase real", GESTOS[int(y_true[idx])],
                  help=GESTOS_DESC[int(y_true[idx])])

    with col_info:
        idx = st.session_state.idx

        # Inferencia en vivo de la ventana visible; el cache cubre el caso sin PyTorch.
        if ensemble is not None:
            probs, latency_ms = infer_live(ensemble, signals, feat48, feat132, idx)
        else:
            probs, latency_ms = probs_all[idx], float(cache["latency_p50"])

        pred = int(predict(probs, theta)[0])
        real = int(y_true[idx])
        correcto = pred == real

        if st.session_state.last_timeline_idx != idx:
            st.session_state.timeline_step += 1
            st.session_state.timeline_history.append({
                "step": st.session_state.timeline_step, "idx": idx,
                "pred": pred, "real": real,
                "conf": float(probs[pred]), "correct": correcto,
            })
            st.session_state.timeline_history = st.session_state.timeline_history[-60:]
            st.session_state.last_timeline_idx = idx

        color = COLORES[pred]
        icono = "✅" if correcto else "❌"
        regla = "argmax" if theta is None else f"umbral theta={theta:.2f}"
        st.markdown(
            f"""
            <div class="tdg-pred-card" style="border-color: {color}55;
                 box-shadow: 0 14px 35px rgba(0,0,0,0.28), 0 0 0 1px {color}33;">
                <div class="tdg-pred-label">Salida del ensemble 4M · {regla}</div>
                <div class="tdg-pred-title" style="color:{color};">{icono} {GESTOS[pred]}</div>
                <div class="tdg-pred-desc">{GESTOS_DESC[pred]}</div>
                <div class="tdg-pred-meta">
                    <span style="color:{UI_MUTED};">Confianza Top-1:</span>
                    <strong style="color:{color};">{probs[pred]*100:.1f}%</strong>
                    &nbsp;·&nbsp;
                    <span style="color:{UI_MUTED};">Etiqueta real:</span>
                    <strong>{GESTOS[real]}</strong>
                    &nbsp;·&nbsp;
                    <span style="color:{UI_MUTED};">Inferencia:</span>
                    <strong>{latency_ms:.1f} ms</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Confianza Top-1", f"{probs[pred]*100:.1f}%")
        k2.metric("Margen sobre Top-2",
                  f"{(np.sort(probs)[-1] - np.sort(probs)[-2])*100:.1f} pp",
                  help="Diferencia entre la clase ganadora y la siguiente. Un margen "
                       "estrecho senala una decision fragil.")
        p95 = float(cache["latency_p95"])
        k3.metric("Latencia p95", f"{p95:.1f} ms",
                  delta=f"{p95 - LATENCY_BUDGET_MS:.0f} ms vs limite",
                  delta_color="inverse",
                  help=f"Percentil 95 sobre {int(cache['latency_runs'])} inferencias tras "
                       f"warm-up. El control protesico natural exige menos de "
                       f"{LATENCY_BUDGET_MS} ms.")

        estado_ok = p95 < LATENCY_BUDGET_MS
        color_estado = UI_SUCCESS if estado_ok else UI_DANGER
        texto_estado = (f"✓ Margen de {LATENCY_BUDGET_MS - p95:.0f} ms sobre el limite de "
                        f"{LATENCY_BUDGET_MS} ms para control protesico"
                        if estado_ok else
                        f"⚠ Latencia por encima del limite de {LATENCY_BUDGET_MS} ms")
        st.markdown(
            f'<div style="background:{color_estado}1a;border:1px solid {color_estado}55;'
            f'border-radius:8px;padding:7px 12px;font-size:0.87rem;font-weight:600;'
            f'color:{color_estado};margin-bottom:14px;">{texto_estado}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="tdg-kicker">Senal, probabilidades y features</p>',
                    unsafe_allow_html=True)

        selected_ch = st.multiselect(
            "Canales EMG visibles:", options=list(range(12)), default=list(range(12)),
            format_func=lambda x: f"Ch{x+1}", key="ch_select",
            help="Aisla canales concretos para reducir el ruido visual de la grafica.",
        )

        col_sig, col_conf = st.columns([3, 2])
        with col_sig:
            st.plotly_chart(fig_emg_window(signals[idx, 0], selected_ch),
                            width="stretch", config=PLOTLY_CONFIG)
        with col_conf:
            st.plotly_chart(fig_confidence_bars(probs, top_n=3),
                            width="stretch", config=PLOTLY_CONFIG)
            with st.expander("Ver distribucion completa (7 clases)"):
                st.plotly_chart(fig_confidence_bars(probs),
                                width="stretch", config=PLOTLY_CONFIG)

        st.plotly_chart(fig_cyborg_features(feat48[idx]),
                        width="stretch", config=PLOTLY_CONFIG)

        st.markdown('<p class="tdg-kicker">Comportamiento temporal</p>', unsafe_allow_html=True)
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            st.plotly_chart(fig_prediction_timeline(st.session_state.timeline_history),
                            width="stretch", config=PLOTLY_CONFIG)
        with hist_col2:
            st.plotly_chart(fig_confidence_trend(st.session_state.timeline_history),
                            width="stretch", config=PLOTLY_CONFIG)

    if st.session_state.play:
        time.sleep({"0.5x": 0.9, "1x": 0.45, "2x": 0.22}[st.session_state.play_speed])
        if st.session_state.idx < n_windows - 1:
            st.session_state.idx += 1
        else:
            st.session_state.play = False
        st.rerun(scope="fragment")


def render_metrics(cache: dict, theta: float | None) -> None:
    """Panel de metricas sobre el conjunto de prueba completo."""
    y_true = cache["labels"]
    probs_all = cache["probs_mean"]
    y_pred = predict(probs_all, theta)

    acc = np.mean(y_pred == y_true) * 100
    bal = balanced_accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    acc_argmax = np.mean(predict(probs_all, None) == y_true) * 100

    st.markdown('<p class="tdg-kicker">Resumen cuantitativo</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{acc:.2f}%",
              delta=None if theta is None else f"{acc - acc_argmax:+.2f}% vs argmax")
    c2.metric("Balanced Accuracy", f"{bal:.2f}%",
              help="Media de los recalls por clase. Corrige el desbalance: el reposo "
                   "supone cerca del 60% de las ventanas.")
    c3.metric("Kappa de Cohen", f"{kappa:.3f}",
              help="Acuerdo corregido por azar. 1.0 es acuerdo perfecto, 0.0 equivale "
                   "a clasificar al azar.")
    c4.metric("Brecha al objetivo", f"{PAPER_BASELINE - acc:.2f} pp",
              help=f"Diferencia frente al {PAPER_BASELINE:.2f}% reportado por el paper "
                   "de referencia sobre el mismo numero de clases.")

    if theta is not None:
        bal_argmax = balanced_accuracy_score(y_true, predict(probs_all, None)) * 100
        st.caption(
            f"⚖️ **Compromiso del umbral.** La regla sube la accuracy global de "
            f"{acc_argmax:.2f}% a {acc:.2f}%, pero baja la balanced accuracy de "
            f"{bal_argmax:.2f}% a {bal:.2f}%. El motivo es que predice reposo siempre que "
            f"P(reposo) supere theta, incluso cuando otra clase puntua mas alto: gana aciertos "
            f"en la clase mayoritaria y pierde recall en los gestos. La accuracy global favorece "
            f"esta regla porque el reposo supone cerca del 60% de las ventanas; para una protesis "
            f"real, la balanced accuracy es el criterio mas honesto."
        )

    st.divider()
    col_cm, col_bar = st.columns([3, 2])
    with col_cm:
        st.plotly_chart(fig_confusion_matrix(y_true, y_pred),
                        width="stretch", config=PLOTLY_CONFIG)
    with col_bar:
        st.markdown("**Metricas por clase**")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(7), zero_division=0
        )
        tabla = pd.DataFrame({
            "Gesto": GESTOS,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "N": support,
        })
        st.dataframe(
            tabla.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"})
                 .background_gradient(subset=["F1"], cmap="Blues", vmin=0, vmax=1),
            hide_index=True, width="stretch",
        )
        st.caption(
            f"Macro-F1: **{f1.mean():.3f}** · Macro-precision: **{precision.mean():.3f}** · "
            f"Macro-recall: **{recall.mean():.3f}**"
        )

        fig_bar = go.Figure(go.Bar(
            y=GESTOS, x=recall * 100, orientation="h", marker_color=COLORES,
            text=[f"{v*100:.1f}%" for v in recall], textposition="outside",
            hovertemplate="%{y}: recall %{x:.1f}%<extra></extra>",
        ))
        fig_bar.update_layout(title="Recall por clase",
                              xaxis=dict(range=[0, 112], title="Recall (%)"))
        st.plotly_chart(apply_plotly_style(fig_bar, height=300), width="stretch",
                        config=PLOTLY_CONFIG)

    st.divider()
    st.subheader("Seleccion del umbral de reposo")
    st.caption(
        "El umbral se ajusto exclusivamente sobre la particion de validacion (repeticion 5). "
        "El conjunto de prueba (repeticion 6) se mantuvo intacto durante la busqueda."
    )
    st.plotly_chart(
        fig_theta_curve(cache["theta_grid"], cache["theta_val_acc"],
                        cache["theta_val_bal"], float(cache["theta_opt"])),
        width="stretch", config=PLOTLY_CONFIG,
    )

    st.divider()
    st.subheader("Analisis de errores biomecanicos")

    resumen = error_summary(y_true, y_pred)
    e1, e2, e3 = st.columns(3)
    e1.metric("Gesto perdido como reposo", f"{resumen['gesto_a_reposo']:.1f}%",
              help="Porcentaje de los errores en los que un gesto real se clasifico como reposo.")
    e2.metric("Reposo tomado por gesto", f"{resumen['reposo_a_gesto']:.1f}%",
              help="Porcentaje de los errores en los que una ventana de reposo se clasifico "
                   "como gesto activo.")
    e3.metric("Confusion entre gestos", f"{resumen['entre_gestos']:.1f}%",
              help="Porcentaje de los errores en los que se confundieron dos gestos distintos.")

    st.markdown(
        f"De los **{resumen['total']}** errores del conjunto de prueba, "
        f"**{resumen['gesto_a_reposo']:.0f}%** corresponden a gestos que el modelo no llega a "
        "detectar y asigna a reposo. Ese es el modo de fallo dominante, y es el que motiva tanto "
        "los pesos de clase durante el entrenamiento como el umbral de reposo evaluado arriba: "
        "el problema no es tanto distinguir un dedo de otro como decidir si hay gesto."
    )

    st.caption(
        "Las tres confusiones con mayor numero de casos fuera de la diagonal. La evidencia "
        "cuantitativa se calcula sobre este mismo conjunto de prueba: amplitud RMS relativa "
        "cuando la confusion involucra al reposo, y correlacion entre perfiles medios de "
        "features cuando enfrenta a dos gestos."
    )

    corr = cache["class_corr"]
    rms = class_mean_rms(cache["feat48"], y_true)
    for rank, err in enumerate(get_top_confusions(y_true, y_pred, top_n=3), start=1):
        st.markdown(
            f"**#{rank} — {GESTOS[err['real']]} clasificado como {GESTOS[err['pred']]}**: "
            f"`{err['count']}` de `{err['total']}` ventanas ({err['pct']:.1f}% de la clase) · "
            f"{confusion_evidence(err, corr, rms)}"
        )
        st.info(anat_text(err["real"], err["pred"]), icon="🔬")

    st.divider()
    export = pd.DataFrame({
        "ventana": np.arange(1, len(y_true) + 1),
        "clase_real": [GESTOS[i] for i in y_true],
        "clase_predicha": [GESTOS[i] for i in y_pred],
        "acierto": y_pred == y_true,
        "confianza": probs_all[np.arange(len(y_true)), y_pred],
    })
    st.download_button(
        "Descargar predicciones (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"predicciones_ensemble4m_{'umbral' if theta else 'argmax'}.csv",
        mime="text/csv",
    )


def render_generalization(loro: dict | None) -> None:
    """Panel de validacion cruzada inter-sujeto."""
    st.markdown('<p class="tdg-kicker">Generalizacion inter-sujeto</p>', unsafe_allow_html=True)

    if loro is None:
        st.warning(
            "No se encontro `checkpoints/loro/loro_cv_summary.json`. Genera la validacion "
            "cruzada con `python -m src.train_loro_cv` (repositorio clasificador-semg) para habilitar este panel."
        )
        return

    st.markdown(
        "Los resultados de las demas pestanas son **intra-sujeto**: se entrena y se evalua sobre "
        "S1 separando por repeticion. Esta pestana responde la pregunta complementaria: **como se "
        "comporta el modelo ante un sujeto que nunca vio**. En cada uno de los 6 folds se deja un "
        "sujeto completo fuera del entrenamiento y se evalua sobre el."
    )

    ens = loro.get("ensemble_4m")
    if ens is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy media (6 folds)", f"{ens['accuracy']['mean']*100:.2f}%",
                  delta=f"± {ens['accuracy']['std']*100:.2f}% entre folds")
        c2.metric("F1 macro", f"{ens['f1_macro']['mean']:.3f}")
        c3.metric("Precision macro", f"{ens['precision_macro']['mean']:.3f}")
        c4.metric("Recall macro", f"{ens['recall_macro']['mean']:.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_loro(loro), width="stretch", config=PLOTLY_CONFIG)
    with col_b:
        st.plotly_chart(fig_loro_folds(loro), width="stretch", config=PLOTLY_CONFIG)

    filas = []
    nombres = {"v10": "v10 CyborgHybrid", "v17": "v17 Features 132D",
               "v18": "v18 Self-Attention", "ensemble_4m": "Ensemble 4M"}
    for key in [k for k in ["v10", "v17", "v18", "ensemble_4m"] if k in loro]:
        m = loro[key]
        filas.append({
            "Modelo": nombres[key],
            "Accuracy": f"{m['accuracy']['mean']*100:.2f}% ± {m['accuracy']['std']*100:.2f}",
            "F1 macro": f"{m['f1_macro']['mean']:.3f} ± {m['f1_macro']['std']:.3f}",
            "Precision macro": f"{m['precision_macro']['mean']:.3f}",
            "Recall macro": f"{m['recall_macro']['mean']:.3f}",
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    if ens is not None and "v10" in loro:
        mejora = (ens["accuracy"]["mean"] - loro["v10"]["accuracy"]["mean"]) * 100
        reduccion = (loro["v10"]["accuracy"]["std"] - ens["accuracy"]["std"]) * 100
        st.info(
            f"El ensemble mejora la media en **{mejora:+.2f} pp** frente a v10 y reduce la "
            f"desviacion entre sujetos en **{reduccion:.2f} pp**. Es el argumento central a favor "
            "del ensemble: no solo acierta mas, sino que su rendimiento depende menos de que "
            "sujeto le toque.",
            icon="📌",
        )


def render_evolution(cache: dict, theta: float | None) -> None:
    """Panel con la historia experimental del proyecto."""
    acc_argmax = np.mean(predict(cache["probs_mean"], None) == cache["labels"]) * 100
    acc_theta = np.mean(
        predict(cache["probs_mean"], float(cache["theta_opt"])) == cache["labels"]
    ) * 100
    best_value = max(acc_argmax, acc_theta)
    best_label = "Ensemble 4M + umbral" if acc_theta >= acc_argmax else "Ensemble 4M"
    live = {"Ensemble 4M": acc_argmax, "Ensemble 4M + umbral": acc_theta}

    st.markdown('<p class="tdg-kicker">Linea de tiempo experimental</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_evolution(best_label, best_value, live),
                    width="stretch", config=PLOTLY_CONFIG)

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown(
            "🔴 **Hito 1 — Caida Focal/Z-Score**  \n"
            "Focal Loss agresivo combinado con normalizacion Z-Score desestabilizo la "
            "convergencia (51.99%). La leccion fue validar cada componente por separado "
            "antes de combinarlos."
        )
    with col_h2:
        st.markdown(
            "🟢 **Hito 2 — Arquitectura Cyborg (v10)**  \n"
            "Primera fusion de CNN 1D + BiLSTM con features manuales de 48D. Mejoro en "
            "1.37 pp sobre la CNN pura y confirmo que RMS, ZC y WL aportan informacion "
            "complementaria a la red."
        )
    with col_h3:
        st.markdown(
            "🟣 **Hito 3 — Multi-sujeto (v13)**  \n"
            "Entrenamiento con S1–S4 y Mixup. No supero a v10 sobre S1 (68.91%), pero "
            "validó la generalizacion inter-sujeto y abrio la via del fine-tuning (v14)."
        )

    st.divider()
    st.subheader("Los 4 modelos del ensemble")
    modelos = [
        ("v10 CyborgHybrid", "CNN 1D + BiLSTM. Arquitectura base.", 71.32, "48D"),
        ("v15 Augmentation", "Igual que v10, con augmentation agresivo.", 70.74, "48D"),
        ("v17 Features 132D", "Features extendidas: MAV, VAR, WAMP, AR(1-4).", 71.89, "132D"),
        ("v18 Self-Attention", "v10 mas atencion multi-cabeza sobre el BiLSTM.", 71.63, "48D"),
    ]
    for col, (nombre, desc, acc_ind, feat) in zip(st.columns(4), modelos):
        with col:
            st.markdown(f"**{nombre}**")
            st.caption(desc)
            st.metric("Accuracy individual", f"{acc_ind:.2f}%")
            st.caption(f"Features manuales: {feat}")

    st.divider()
    st.info(
        f"**Principio del ensemble.** Cada modelo comete errores distintos sobre la misma senal, "
        f"de modo que promediar sus probabilidades softmax corrige parte de los fallos "
        f"individuales. El ensemble uniforme alcanza **{acc_argmax:.2f}%** y con el umbral de "
        f"reposo **{acc_theta:.2f}%**, frente al **{PAPER_BASELINE:.2f}%** del paper de "
        f"referencia: una brecha de **{PAPER_BASELINE - best_value:.2f} pp** que la pestana de "
        f"generalizacion ayuda a contextualizar."
    )


def render_methodology(cache: dict) -> None:
    """Panel con el pipeline de procesamiento y la arquitectura."""
    st.markdown('<p class="tdg-kicker">Cadena de procesamiento</p>', unsafe_allow_html=True)
    st.subheader("De la senal cruda a la ventana clasificable")

    pasos = [
        ("01", "Adquisicion", "12 electrodos Delsys sobre el miembro residual, muestreados "
                              "a 2000 Hz (NinaPro DB3)."),
        ("02", "Submuestreo", "Factor 4 hasta 500 Hz. Conserva el contenido util del EMG de "
                              "superficie y reduce el coste computacional."),
        ("03", "Filtrado", "Pasa-banda Butterworth de orden 4 entre 20 y 249 Hz, aplicado con "
                           "filtfilt para no introducir desfase."),
        ("04", "Normalizacion", "Z-score por canal y por sujeto, de modo que la escala "
                                "individual de cada electrodo no sesgue el modelo."),
        ("05", "Ventaneo", "Ventanas de 250 ms (125 muestras) con paso de 50 ms. Se descartan "
                           "las ventanas de transicion entre gestos."),
        ("06", "Features", "Canal RMS anadido como 13.º y vector manual de 48D o 132D "
                           "calculado por canal."),
    ]
    for row in range(2):
        for col, (n, titulo, desc) in zip(st.columns(3), pasos[row * 3:(row + 1) * 3]):
            with col:
                st.markdown(
                    f'<div class="tdg-step"><div class="tdg-step-n">PASO {n}</div>'
                    f'<div class="tdg-step-t">{titulo}</div>'
                    f'<div class="tdg-step-d">{desc}</div></div>',
                    unsafe_allow_html=True,
                )
        st.write("")

    st.divider()
    st.subheader("Arquitectura CyborgHybrid")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            """
            La red combina dos vias que ven la misma ventana desde perspectivas distintas:

            **Rama temporal.** Tres capas convolucionales 1D (13 → 64 → 128 → 128 canales)
            extraen patrones locales de activacion, y un BiLSTM de 2 capas con 128 unidades
            ocultas modela la dependencia temporal en ambos sentidos. La salida agrupada
            produce un vector de 256D.

            **Rama de features.** El vector manual de 48D (RMS, ZC, SSC y WL por canal) pasa
            por un MLP que lo proyecta a 64D. Son descriptores clasicos de la literatura de
            EMG, con decadas de respaldo experimental.

            **Fusion.** Ambas ramas se concatenan en 320D, se reducen a 128D y se proyectan
            sobre las 7 clases. La rama manual actua como regularizador: aporta informacion
            robusta que la red no necesita aprender desde cero con datos limitados.
            """
        )
    with col_b:
        st.markdown(
            """
            **Parametros del entrenamiento**

            | Parametro | Valor |
            |---|---|
            | Frecuencia objetivo | 500 Hz |
            | Ventana | 125 muestras (250 ms) |
            | Paso | 25 muestras (50 ms) |
            | Batch | 64 |
            | Learning rate | 5e-4 |
            | Optimizador | AdamW (wd 1e-4) |
            | Pesos de clase | inverso de la raiz |
            | Early stopping | paciencia 15 |
            | Semilla | 42 |
            """
        )
        st.caption(
            "La particion es por repeticion — entrenamiento 1-4, validacion 5, prueba 6 — "
            "de modo que ninguna ventana del conjunto de prueba comparte repeticion con el "
            "entrenamiento."
        )

    st.divider()
    st.subheader("Reproducibilidad de esta demo")
    st.markdown(
        f"""
        Las {len(cache['labels']):,} ventanas de prueba, sus features y las probabilidades de los
        cuatro modelos se generan con `python -m src.build_demo_cache` (repositorio clasificador-semg), que las persiste en
        `data/demo_cache.npz`. La demo no vuelve a tocar los archivos `.mat` originales, de modo
        que arranca en segundos y produce exactamente los mismos numeros en cualquier equipo.
        La inferencia de la ventana visible si se ejecuta en vivo sobre los cuatro modelos.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# Aplicacion
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Punto de entrada de la aplicacion Streamlit."""
    st.set_page_config(
        page_title="Clasificador EMG — Trabajo de Grado UV",
        page_icon="🦾",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    os.chdir(ROOT)
    inject_demo_css()

    st.markdown(
        """
        <div class="tdg-hero">
            <div class="tdg-hero-badge">Demo tecnica · Ensemble 4M · Inferencia en CPU</div>
            <div class="tdg-hero-title">🦾 Clasificacion de gestos EMG con redes hibridas</div>
            <p class="tdg-hero-sub">
                Trabajo de Grado · Escuela de Ingenieria Electrica y Electronica ·
                Universidad del Valle · NinaPro DB3 · Ventanas de 250&nbsp;ms a 500&nbsp;Hz
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        cache = load_cache(CACHE_VERSION)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # cache corrupto o incompleto
        st.error(f"No se pudo leer el cache de la demo: {exc}")
        st.stop()

    ensemble = load_ensemble()
    loro = load_loro()

    st.markdown(
        f'<div class="tdg-prov">NinaPro DB3 · sujeto S{int(cache["subject"])} · '
        f'conjunto de prueba = repeticion {int(cache["test_rep"])} '
        f'({len(cache["labels"]):,} ventanas) · evaluacion intra-sujeto · '
        f'7 clases · ensemble de 4 modelos</div>',
        unsafe_allow_html=True,
    )

    theta = render_controls(cache, ensemble)

    tab_sim, tab_metrics, tab_loro, tab_evol, tab_method = st.tabs([
        "🎮 Simulacion en vivo",
        "📊 Metricas del conjunto de prueba",
        "🌐 Generalizacion inter-sujeto",
        "📈 Evolucion de versiones",
        "🔬 Metodologia",
    ])

    with tab_sim:
        render_simulation(cache, ensemble, theta)
    with tab_metrics:
        render_metrics(cache, theta)
    with tab_loro:
        render_generalization(loro)
    with tab_evol:
        render_evolution(cache, theta)
    with tab_method:
        render_methodology(cache)


if __name__ == "__main__":
    main()
