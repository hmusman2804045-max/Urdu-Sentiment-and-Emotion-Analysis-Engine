import gradio as gr
from predictor import SentimentEmotionPredictor

# ── Load models once at startup ──────────────────────────────────────────────
print("Initialising Urdu Sentiment & Emotion Engine…")
engine = SentimentEmotionPredictor()
print("Engine ready.")

# ── Emoji / colour maps ───────────────────────────────────────────────────────
SENTIMENT_EMOJI = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
EMOTION_EMOJI   = {"Joy": "🎉", "Anger": "😡", "Fear": "😨", "Sadness": "😢"}

SENTIMENT_COLOR = {
    "Positive": "#22c55e",
    "Negative": "#ef4444",
    "Neutral":  "#facc15",
}
EMOTION_COLOR = {
    "Joy":     "#f59e0b",
    "Anger":   "#ef4444",
    "Fear":    "#8b5cf6",
    "Sadness": "#3b82f6",
}

# ── Example inputs ─────────────────────────────────────────────────────────────
EXAMPLES = [
    ["آج کا دن بہت اچھا ہے، بہت خوشی ہوئی"],
    ["mujhe bohat gussa aa raha hai is cheez par"],
    ["یہ صورتحال بہت خطرناک اور ڈراؤنی ہے"],
    ["Aaj mera dil bohat udaas hai, kuch bhi acha nahi lag raha"],
    ["بالکل ٹھیک ہے، کوئی خاص بات نہیں"],
    ["Yeh sab dekh kar dil khush ho gaya, wah wah!"],
]


def build_attention_html(attention_list):
    if not attention_list:
        return "<p style='color:#9ca3af;font-size:0.85rem'>No attention data.</p>"
    max_score = max(a["score"] for a in attention_list) or 1.0
    html = "<div style='display:flex;flex-wrap:wrap;gap:6px;padding:8px 0;'>"
    for item in attention_list:
        intensity = item["score"] / max_score
        alpha     = 0.15 + intensity * 0.75
        font_w    = 400 + int(intensity * 300)
        html += (
            f"<span style='background:rgba(139,92,246,{alpha:.2f});color:#e9d5ff;"
            f"padding:3px 8px;border-radius:12px;font-size:0.9rem;"
            f"font-weight:{font_w};border:1px solid rgba(139,92,246,0.3);'>"
            f"{item['word']}</span>"
        )
    html += "</div>"
    return html


def build_bar(label, score, color):
    pct = round(score * 100, 1)
    return (
        f"<div style='margin-bottom:8px;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;"
        f"color:#d1d5db;margin-bottom:3px;'><span>{label}</span><span>{pct}%</span></div>"
        f"<div style='background:#1f2937;border-radius:999px;height:8px;overflow:hidden;'>"
        f"<div style='width:{pct}%;background:{color};height:100%;border-radius:999px;"
        f"transition:width 0.6s ease;'></div></div></div>"
    )


def analyse(text):
    if not text or not text.strip():
        return (
            "<p style='color:#ef4444'>Please enter some Urdu or Roman Urdu text.</p>",
            "", "", "",
        )

    result = engine.predict(text)

    if "error" in result:
        return (f"<p style='color:#ef4444'>{result['error']}</p>", "", "", "")

    sentiment = result["sentiment"]
    emotion   = result["emotion"]
    s_scores  = result["sentiment_scores"]
    e_scores  = result["emotion_scores"]
    attention = result["attention"]

    s_emoji = SENTIMENT_EMOJI.get(sentiment, "")
    e_emoji = EMOTION_EMOJI.get(emotion, "")
    s_color = SENTIMENT_COLOR.get(sentiment, "#6b7280")
    e_color = EMOTION_COLOR.get(emotion, "#6b7280")

    result_html = f"""
<div style='background:linear-gradient(135deg,#1e1b4b 0%,#111827 100%);
border:1px solid rgba(139,92,246,0.35);border-radius:16px;padding:20px 24px;
font-family:Inter,sans-serif;'>
  <div style='display:flex;gap:16px;flex-wrap:wrap;'>
    <div style='flex:1;min-width:140px;background:rgba(0,0,0,0.3);
    border:2px solid {s_color};border-radius:12px;padding:14px 18px;text-align:center;'>
      <div style='font-size:2rem;'>{s_emoji}</div>
      <div style='font-size:0.72rem;letter-spacing:0.1em;color:#9ca3af;margin:4px 0 2px;'>SENTIMENT</div>
      <div style='font-size:1.25rem;font-weight:700;color:{s_color};'>{sentiment}</div>
    </div>
    <div style='flex:1;min-width:140px;background:rgba(0,0,0,0.3);
    border:2px solid {e_color};border-radius:12px;padding:14px 18px;text-align:center;'>
      <div style='font-size:2rem;'>{e_emoji}</div>
      <div style='font-size:0.72rem;letter-spacing:0.1em;color:#9ca3af;margin:4px 0 2px;'>EMOTION</div>
      <div style='font-size:1.25rem;font-weight:700;color:{e_color};'>{emotion}</div>
    </div>
  </div>
</div>
"""

    s_bars_html = "<div style='padding:4px 0;'>"
    for lbl, sc in s_scores.items():
        s_bars_html += build_bar(lbl, sc, SENTIMENT_COLOR.get(lbl, "#6b7280"))
    s_bars_html += "</div>"

    e_bars_html = "<div style='padding:4px 0;'>"
    for lbl, sc in e_scores.items():
        e_bars_html += build_bar(lbl, sc, EMOTION_COLOR.get(lbl, "#6b7280"))
    e_bars_html += "</div>"

    attn_html = build_attention_html(attention)

    return result_html, s_bars_html, e_bars_html, attn_html


CSS = """
body, .gradio-container {
    background: #0f0c29 !important;
    font-family: 'Inter', sans-serif !important;
}
#header-banner {
    background: linear-gradient(135deg,#1a0533 0%,#0f172a 50%,#0c1445 100%);
    border-bottom: 1px solid rgba(139,92,246,0.3);
    padding: 28px 24px 18px;
    text-align: center;
    border-radius: 16px 16px 0 0;
    margin-bottom: 4px;
}
#header-banner h1 {
    font-size: clamp(1.4rem, 4vw, 2rem);
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
    letter-spacing: -0.02em;
}
#header-banner p { color: #94a3b8; font-size: 0.9rem; margin: 0; }
#input-box textarea {
    background: #1e1b4b !important;
    border: 1.5px solid rgba(139,92,246,0.4) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding: 14px !important;
}
#input-box textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}
#analyse-btn {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: #fff !important;
    padding: 10px 0 !important;
    transition: opacity 0.2s !important;
}
#analyse-btn:hover { opacity: 0.88 !important; }
#clear-btn {
    background: rgba(31,41,55,0.8) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    color: #9ca3af !important;
}
.section-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #7c3aed;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.output-panel {
    background: rgba(17,24,39,0.85) !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
#footer {
    text-align: center;
    color: #4b5563;
    font-size: 0.78rem;
    margin-top: 16px;
    padding: 12px 0 4px;
    border-top: 1px solid rgba(139,92,246,0.15);
}
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
    title="Urdu Sentiment & Emotion Engine",
) as demo:

    gr.HTML("""
    <div id="header-banner">
      <h1>🇵🇰 Urdu Sentiment &amp; Emotion Analysis Engine</h1>
      <p>XLM-RoBERTa fine-tuned on Urdu · Roman Urdu · Mixed language text</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            gr.HTML("<p class='section-label'>✍️ Enter Text</p>")
            text_input = gr.Textbox(
                placeholder="اردو یا Roman Urdu میں لکھیں…\nYa Roman Urdu mein likhein…",
                lines=5,
                max_lines=10,
                show_label=False,
                elem_id="input-box",
            )
            with gr.Row():
                analyse_btn = gr.Button("🔍 Analyse", variant="primary", elem_id="analyse-btn")
                clear_btn   = gr.Button("✕ Clear",   variant="secondary", elem_id="clear-btn")

            gr.HTML("<p class='section-label' style='margin-top:18px;'>💡 Try an Example</p>")
            gr.Examples(examples=EXAMPLES, inputs=text_input, label="")

        with gr.Column(scale=5):
            gr.HTML("<p class='section-label'>🎯 Prediction</p>")
            result_out = gr.HTML(elem_classes=["output-panel"])

            with gr.Row():
                with gr.Column():
                    gr.HTML("<p class='section-label' style='margin-top:14px;'>📊 Sentiment Confidence</p>")
                    sent_bars = gr.HTML(elem_classes=["output-panel"])
                with gr.Column():
                    gr.HTML("<p class='section-label' style='margin-top:14px;'>📊 Emotion Confidence</p>")
                    emot_bars = gr.HTML(elem_classes=["output-panel"])

            gr.HTML("<p class='section-label' style='margin-top:14px;'>🔦 Word Attention Highlights</p>")
            attn_out = gr.HTML(elem_classes=["output-panel"])

    gr.HTML("""
    <div id="footer">
      Powered by <strong>XLM-RoBERTa</strong> · Fine-tuned by <strong>Muhammad Usman</strong> ·
      <a href="https://github.com/hmusman2804045-max/Urdu-Sentiment-and-Emotion-Analysis-Engine"
         style="color:#7c3aed;" target="_blank">GitHub ↗</a>
    </div>
    """)

    analyse_btn.click(fn=analyse, inputs=text_input,
                      outputs=[result_out, sent_bars, emot_bars, attn_out])
    text_input.submit(fn=analyse, inputs=text_input,
                      outputs=[result_out, sent_bars, emot_bars, attn_out])
    clear_btn.click(fn=lambda: ("", "", "", ""), inputs=None,
                    outputs=[result_out, sent_bars, emot_bars, attn_out])

if __name__ == "__main__":
    demo.launch()
