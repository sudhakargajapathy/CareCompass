"""Hearth design system for CareCompass.

Light, warm palette adapted from the Care Navigator design prototype's
:root tokens.
Provides the CSS layer injected into Streamlit.
"""

# Design tokens (hex values from the prototype)
HEARTH = {
    "bg": "#FBF7F1",
    "bone_50": "#F5EFE5",
    "bone_100": "#EDE4D4",
    "bone_200": "#E0D3BE",
    "bone_300": "#C9B89C",
    "clay_400": "#B08968",
    "clay_600": "#7D5A3F",
    "clay_800": "#4A3525",
    "accent_50": "#F7EEDF",
    "accent_100": "#F0DFC0",
    "accent_200": "#E4CA9B",
    "accent_400": "#C8A26C",
    "accent_500": "#B8894A",
    "accent_700": "#8A6434",
    "moss_100": "#E4EADC",
    "moss_500": "#7A9465",
    "moss_600": "#5F7A4C",
    "moss_700": "#4D6240",
    "rust_100": "#F3E4DA",
    "rust_400": "#C08460",
    "rust_600": "#9A5E3F",
    "ink": "#2B2018",
    "ink_soft": "#6B5B4B",
}

HEARTH_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Inter:wght@400;500;600;700&display=swap');

/* ---- Typography ---- */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    color: {HEARTH["ink"]};
}}
.stApp p, .stApp label, .stApp li, .stApp td, .stApp th,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
}}
.stApp h1, .stApp h2, .stApp h3 {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: {HEARTH["ink"]};
}}
.stApp h1 {{ font-size: 2.2rem; }}
.stApp h2 {{ font-size: 1.6rem; }}

/* ---- Header hero ---- */
.cc-hero {{ padding: 0.25rem 0 0.5rem 0; }}
.cc-hero-eyebrow {{
    display: inline-block;
    background: {HEARTH["accent_100"]};
    color: {HEARTH["clay_800"]};
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}}
.cc-hero-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.7rem;
    font-weight: 500;
    letter-spacing: -0.015em;
    line-height: 1.05;
    color: {HEARTH["ink"]};
    margin: 0;
}}
.cc-hero-sub {{
    color: {HEARTH["ink_soft"]};
    font-size: 1.05rem;
    margin-top: 0.35rem;
}}

/* ---- Buttons ---- */
.stButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {{
    border-radius: 999px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: transform 0.05s ease;
}}
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active {{
    transform: scale(0.98);
}}
[data-testid="stBaseButton-primary"] {{
    background: {HEARTH["clay_800"]};
    color: {HEARTH["bone_50"]};
    border: none;
}}
[data-testid="stBaseButton-primary"]:hover {{ background: {HEARTH["clay_600"]}; }}
[data-testid="stBaseButton-secondary"] {{
    background: {HEARTH["bone_100"]};
    color: {HEARTH["ink"]};
    border: 1px solid {HEARTH["bone_300"]};
}}

/* ---- Search form card ---- */
[data-testid="stForm"] {{
    background: {HEARTH["bone_50"]};
    border: 1px solid {HEARTH["bone_200"]};
    border-radius: 24px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 10px 30px -8px rgba(80, 55, 40, 0.08);
}}

/* ---- Inputs ---- */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{
    background: {HEARTH["bg"]};
    border-color: {HEARTH["bone_300"]} !important;
    border-radius: 12px !important;
}}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{
    background: {HEARTH["bg"]};
    color: {HEARTH["ink"]};
}}

/* ---- Segmented control (weight pickers) ----
   Streamlit 1.59 DOM: stButtonGroup > div[data-orientation] > button[data-variant="segmented_control"],
   selection carried on aria-checked. */
[data-testid="stButtonGroup"] div[data-orientation="horizontal"]:has(button[data-variant="segmented_control"]) {{
    background: {HEARTH["bone_100"]};
    border: 1px solid {HEARTH["bone_200"]};
    border-radius: 999px;
    padding: 4px;
    gap: 2px;
    width: fit-content;
}}
button[data-variant="segmented_control"] {{
    border-radius: 999px !important;
    border: none !important;
    background: transparent;
    color: {HEARTH["ink"]};
    font-weight: 400;
    padding: 2px 14px;
    min-height: 1.9rem;
}}
button[data-variant="segmented_control"]:hover {{
    color: {HEARTH["clay_800"]};
    background: rgba(251, 247, 241, 0.6);
}}
button[data-variant="segmented_control"][aria-checked="true"] {{
    background: {HEARTH["bg"]};
    color: {HEARTH["ink"]};
    font-weight: 500;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
}}
button[data-variant="segmented_control"][aria-checked="true"]:hover {{
    background: {HEARTH["bg"]};
}}

/* ---- Expanders ---- */
[data-testid="stExpander"] details {{
    background: {HEARTH["bg"]};
    border: 1px solid {HEARTH["bone_200"]};
    border-radius: 14px;
}}
[data-testid="stExpander"] summary {{
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: {HEARTH["bone_50"]};
    border-right: 1px solid {HEARTH["bone_200"]};
}}

/* ---- Metrics ---- */
[data-testid="stMetric"] {{
    background: {HEARTH["bone_50"]};
    border: 1px solid {HEARTH["bone_200"]};
    border-radius: 16px;
    padding: 12px 16px;
}}

/* ---- Alerts ---- */
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---- Provider cards ---- */
.cc-card {{
    background: {HEARTH["bg"]};
    border: 1px solid {HEARTH["bone_200"]};
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 14px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}}
.cc-card:hover {{
    border-color: {HEARTH["clay_400"]};
    transform: translateY(-1px);
}}
.cc-card-top {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
}}
.cc-rank {{
    color: {HEARTH["accent_500"]};
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.15rem;
    margin-right: 6px;
}}
.cc-name {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: {HEARTH["ink"]};
    margin: 0;
}}
.cc-spec {{ color: {HEARTH["ink_soft"]}; font-size: 0.92rem; margin-top: 2px; }}
.cc-meta {{
    color: {HEARTH["ink_soft"]};
    font-size: 0.86rem;
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
}}
.cc-stars {{ color: {HEARTH["accent_700"]}; letter-spacing: 1px; }}
.cc-why {{
    background: {HEARTH["accent_50"]};
    border-left: 3px solid {HEARTH["accent_400"]};
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 12px;
    font-size: 0.9rem;
    color: {HEARTH["ink"]};
}}
.cc-why b {{ color: {HEARTH["clay_600"]}; }}
.cc-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }}
.cc-chip {{
    background: {HEARTH["bone_100"]};
    color: {HEARTH["ink"]};
    font-size: 0.76rem;
    font-weight: 500;
    padding: 4px 11px;
    border-radius: 999px;
}}
.cc-chip--moss {{ background: {HEARTH["moss_100"]}; color: {HEARTH["moss_700"]}; }}
.cc-chip--rust {{ background: {HEARTH["rust_100"]}; color: {HEARTH["rust_600"]}; }}
.cc-ring-wrap {{ text-align: center; flex-shrink: 0; }}
.cc-ring-label {{
    color: {HEARTH["ink_soft"]};
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-top: 2px;
}}

/* ---- Cost card ---- */
.cc-cost-card {{
    background: {HEARTH["bone_50"]};
    border: 1px solid {HEARTH["bone_200"]};
    border-radius: 20px;
    padding: 18px 24px;
    margin: 6px 0 18px 0;
}}
.cc-cost-head {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
}}
.cc-cost-total {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.1rem;
    color: {HEARTH["clay_800"]};
}}
.cc-cost-sub {{ color: {HEARTH["ink_soft"]}; font-size: 0.9rem; }}
.cc-cost-table {{ width: 100%; margin-top: 10px; font-size: 0.84rem; border-collapse: collapse; }}
.cc-cost-table td {{
    padding: 5px 10px 5px 0;
    color: {HEARTH["ink"]};
    border-bottom: 1px solid {HEARTH["bone_100"]};
}}
.cc-cost-table td:first-child {{ color: {HEARTH["ink_soft"]}; }}
.cc-cost-table td:last-child {{ text-align: right; font-weight: 500; }}
.cc-cost-note {{ color: {HEARTH["ink_soft"]}; font-size: 0.76rem; margin-top: 10px; }}
.cc-step-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}

/* ---- Responsible AI panel ---- */
.cc-panel-head {{
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}}
.cc-panel-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.3rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: {HEARTH["ink"]};
}}
.cc-trust-grid {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
}}
.cc-trust-item {{
    background: {HEARTH["bone_50"]};
    border: 1px solid {HEARTH["bone_100"]};
    border-radius: 14px;
    padding: 10px 14px;
    flex: 1;
    min-width: 160px;
}}
.cc-trust-label {{
    color: {HEARTH["ink_soft"]};
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}}
.cc-trust-value {{
    font-size: 0.92rem;
    font-weight: 500;
    color: {HEARTH["ink"]};
    margin-top: 4px;
}}

/* ---- Score-composition bars ---- */
.cc-bar-row {{ display: flex; align-items: center; gap: 10px; margin: 5px 0; }}
.cc-bar-label {{ width: 92px; font-size: 0.8rem; color: {HEARTH["ink_soft"]}; flex-shrink: 0; }}
.cc-bar-track {{
    flex: 1;
    height: 8px;
    background: {HEARTH["bone_100"]};
    border-radius: 999px;
    overflow: hidden;
}}
.cc-bar-fill {{
    display: block;
    height: 100%;
    background: {HEARTH["accent_500"]};
    border-radius: 999px;
}}
.cc-bar-val {{
    font-size: 0.76rem;
    color: {HEARTH["ink_soft"]};
    min-width: 128px;
    text-align: right;
    flex-shrink: 0;
}}
.cc-bar-quote {{
    color: {HEARTH["ink_soft"]};
    font-size: 0.76rem;
    font-style: italic;
    margin: 0 0 6px 180px;
}}
/* A band the judge scored neutral because it found nothing — same slot and
   alignment as a citation, but upright and dimmer so it never reads as one. */
.cc-bar-nodata {{
    font-style: normal;
    opacity: 0.75;
}}

/* ---- Pipeline strip (How it works) ---- */
.cc-pipe {{ display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; margin: 8px 0 6px 0; }}
.cc-pipe-step {{
    background: {HEARTH["bone_50"]};
    border: 1px solid {HEARTH["bone_200"]};
    border-radius: 12px;
    padding: 8px 12px;
}}
.cc-pipe-name {{ font-size: 0.82rem; font-weight: 600; color: {HEARTH["ink"]}; }}
.cc-pipe-model {{ font-size: 0.74rem; color: {HEARTH["ink_soft"]}; margin-top: 2px; }}
.cc-pipe-arrow {{ align-self: center; color: {HEARTH["bone_300"]}; font-size: 1.1rem; }}

/* ---- Disclaimer footer ---- */
.cc-footer {{
    border-top: 1px solid {HEARTH["bone_200"]};
    margin-top: 30px;
    padding: 14px 2px 6px 2px;
    color: {HEARTH["ink_soft"]};
    font-size: 0.78rem;
    line-height: 1.5;
}}
.cc-footer a {{ color: {HEARTH["clay_600"]}; }}
"""


def inject_theme() -> None:
    """Inject the Hearth CSS layer into the current Streamlit page."""
    import streamlit as st

    st.markdown(f"<style>{HEARTH_CSS}</style>", unsafe_allow_html=True)


