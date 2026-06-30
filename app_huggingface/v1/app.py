import os
import html
from PIL import Image
import gradio as gr
from transformers import pipeline

# =========================
# Modelo
# =========================
MODEL_ID = os.getenv("MODEL_ID", "Antonio27ve/convnext-plantvillage-equipo07")

classifier = pipeline(
    task="image-classification",
    model=MODEL_ID
)

# =========================
# Traducciones
# =========================
CROP_TRANSLATIONS = {
    "Apple": "Manzana",
    "Blueberry": "Arándano",
    "Cherry (including sour)": "Cereza",
    "Corn (maize)": "Maíz",
    "Grape": "Uva",
    "Orange": "Naranja",
    "Peach": "Durazno",
    "Pepper, bell": "Pimiento",
    "Potato": "Papa",
    "Raspberry": "Frambuesa",
    "Soybean": "Soya",
    "Squash": "Calabaza",
    "Strawberry": "Fresa",
    "Tomato": "Tomate",
}

DISEASE_TRANSLATIONS = {
    "Apple scab": "Sarna del manzano",
    "Black rot": "Podredumbre negra",
    "Cedar apple rust": "Roya del manzano",
    "healthy": "Saludable",
    "Powdery mildew": "Oídio",
    "Cercospora leaf spot Gray leaf spot": "Mancha foliar por Cercospora / Mancha gris",
    "Common rust": "Roya común",
    "Northern Leaf Blight": "Tizón foliar del norte",
    "Esca (Black Measles)": "Esca",
    "Leaf blight (Isariopsis Leaf Spot)": "Tizón foliar",
    "Haunglongbing (Citrus greening)": "Huanglongbing",
    "Bacterial spot": "Mancha bacteriana",
    "Early blight": "Tizón temprano",
    "Late blight": "Tizón tardío",
    "Leaf scorch": "Quemado de la hoja",
    "Leaf Mold": "Moho de la hoja",
    "Septoria leaf spot": "Mancha foliar por Septoria",
    "Spider mites Two-spotted spider mite": "Ácaros / Araña roja de dos puntos",
    "Target Spot": "Mancha diana",
    "Tomato Yellow Leaf Curl Virus": "Virus del rizado amarillo del tomate",
    "Tomato mosaic virus": "Virus del mosaico del tomate",
}

# =========================
# Procesamiento de etiquetas
# =========================
def parse_label(label: str):
    """
    Convierte etiquetas tipo:
    Tomato___Early_blight
    Potato___healthy

    en:
    Planta: Tomate
    Diagnóstico: Tizón temprano / Saludable
    """
    raw = label.strip()

    if "___" in raw:
        crop_raw, disease_raw = raw.split("___", 1)
    elif " - " in raw:
        crop_raw, disease_raw = raw.split(" - ", 1)
    else:
        crop_raw, disease_raw = "Desconocido", raw

    crop_raw = crop_raw.replace("_", " ").strip()
    disease_raw = disease_raw.replace("_", " ").strip()

    cultivo_es = CROP_TRANSLATIONS.get(crop_raw, crop_raw)
    diagnostico_es = DISEASE_TRANSLATIONS.get(disease_raw, disease_raw)

    if disease_raw.lower() == "healthy":
        estado = "Hoja sana"
        tipo = "healthy"
    else:
        estado = "Enfermedad detectada"
        tipo = "diseased"

    return cultivo_es, diagnostico_es, estado, tipo


# =========================
# Tarjetas HTML
# =========================
def placeholder_card():
    return """
    <div class="result-card neutral-card">
        <div class="result-badge neutral-badge">🌿 LISTO PARA ANALIZAR</div>
        <div class="result-title">Esperando imagen</div>
        <div class="result-subtitle">
            Sube una imagen de una hoja y presiona <b>Clasificar hoja</b>
            para ver el diagnóstico principal.
        </div>
    </div>
    """


def result_card_html(cultivo, diagnostico, estado, tipo):
    cultivo = html.escape(cultivo)
    diagnostico = html.escape(diagnostico)
    estado = html.escape(estado)

    if tipo == "healthy":
        card_class = "healthy-card"
        badge_class = "healthy-badge"
        badge_text = "✅ HOJA SANA"
    else:
        card_class = "diseased-card"
        badge_class = "diseased-badge"
        badge_text = "⚠️ ENFERMEDAD DETECTADA"

    return f"""
    <div class="result-card {card_class}">
        <div class="result-badge {badge_class}">{badge_text}</div>

        <div class="result-kv">
            <span class="kv-label">Planta</span>
            <span class="kv-value">{cultivo}</span>
        </div>

        <div class="result-kv">
            <span class="kv-label">Estado</span>
            <span class="kv-value">{estado}</span>
        </div>

        <div class="diagnosis-box">
            <div class="diagnosis-label">Diagnóstico principal</div>
            <div class="diagnosis-value">{diagnostico}</div>
        </div>

        <div class="result-footer">
            Resultado generado por un modelo ConvNeXt Tiny entrenado con PlantVillage.
        </div>
    </div>
    """


# =========================
# Predicción
# =========================
def predict(image):
    if image is None:
        return placeholder_card()

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    outputs = classifier(image, top_k=1)
    top_pred = outputs[0]

    cultivo, diagnostico, estado, tipo = parse_label(top_pred["label"])
    return result_card_html(cultivo, diagnostico, estado, tipo)


def clear_interface():
    return None, placeholder_card()


# =========================
# Estilos
# =========================
css = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding-top: 10px !important;
    padding-bottom: 30px !important;
}

footer {
    display: none !important;
}

.main-banner {
    background: linear-gradient(135deg, #0f172a, #1d4ed8, #0f172a);
    color: white;
    border: 1px solid #334155;
    border-radius: 24px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 16px 38px rgba(0,0,0,0.28);
}

.main-banner h1 {
    margin: 0 0 10px 0;
    font-size: 36px;
    line-height: 1.15;
}

.main-banner p {
    margin: 0;
    font-size: 17px;
    color: #dbeafe;
    line-height: 1.6;
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 18px;
}

.info-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 16px;
    padding: 16px 18px;
}

.info-card-title {
    font-weight: 800;
    font-size: 15px;
    margin-bottom: 8px;
}

.info-card-text {
    font-size: 14px;
    line-height: 1.55;
    color: #e5eefc;
}

.result-card {
    min-height: 500px;
    border-radius: 24px;
    padding: 28px;
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 16px 34px rgba(0,0,0,0.25);
}

.neutral-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    align-items: center;
    text-align: center;
}

.healthy-card {
    background: linear-gradient(135deg, #064e3b, #15803d, #064e3b);
    border: 1px solid #4ade80;
}

.diseased-card {
    background: linear-gradient(135deg, #7f1d1d, #b91c1c, #7f1d1d);
    border: 1px solid #fca5a5;
}

.result-badge {
    display: inline-block;
    width: fit-content;
    padding: 9px 15px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 13px;
    margin-bottom: 22px;
    letter-spacing: 0.3px;
}

.neutral-badge {
    background: #dbeafe;
    color: #1e3a8a;
}

.healthy-badge {
    background: #dcfce7;
    color: #166534;
}

.diseased-badge {
    background: #fee2e2;
    color: #991b1b;
}

.result-title {
    font-size: 30px;
    font-weight: 850;
    margin-bottom: 12px;
}

.result-subtitle {
    font-size: 16px;
    color: #d1d5db;
    max-width: 440px;
    line-height: 1.65;
}

.result-kv {
    margin-bottom: 16px;
}

.kv-label {
    display: block;
    font-size: 14px;
    opacity: 0.88;
    margin-bottom: 5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kv-value {
    display: block;
    font-size: 25px;
    font-weight: 850;
}

.diagnosis-box {
    margin-top: 18px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 18px;
    padding: 20px;
}

.diagnosis-label {
    font-size: 14px;
    font-weight: 750;
    opacity: 0.92;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.diagnosis-value {
    font-size: 34px;
    font-weight: 900;
    line-height: 1.2;
}

.result-footer {
    margin-top: 20px;
    font-size: 14px;
    opacity: 0.88;
}

.recommend-box {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 18px;
    padding: 18px 22px;
    color: #e5e7eb;
    margin-top: 14px;
}

.recommend-box h3 {
    margin-top: 0;
    margin-bottom: 10px;
}

.recommend-box ul {
    margin-bottom: 0;
}

@media (max-width: 900px) {
    .info-grid {
        grid-template-columns: 1fr;
    }

    .main-banner h1 {
        font-size: 28px;
    }

    .diagnosis-value {
        font-size: 28px;
    }
}
"""

# =========================
# Interfaz
# =========================
with gr.Blocks(css=css, title="Clasificador de enfermedades en hojas de plantas") as demo:

    gr.HTML(
        """
        <div class="main-banner">
            <h1>🌿 Clasificador de enfermedades en hojas de plantas</h1>
            <p>
                Demo basada en un modelo <b>ConvNeXt Tiny</b> entrenado con <b>PlantVillage</b>.
                Sube una imagen de una hoja y el sistema mostrará el <b>diagnóstico principal</b> en español.
            </p>

            <div class="info-grid">
                <div class="info-card">
                    <div class="info-card-title">📌 Cultivos para los que el modelo es más confiable</div>
                    <div class="info-card-text">
                        Manzana, arándano, cereza, maíz, uva, naranja, durazno, pimiento, papa,
                        frambuesa, soya, calabaza, fresa y tomate.
                    </div>
                </div>

                <div class="info-card">
                    <div class="info-card-title">🧠 Alcance del modelo</div>
                    <div class="info-card-text">
                        Puedes subir hojas de cualquier planta. Si la planta no pertenece a los cultivos
                        del modelo, el resultado debe interpretarse como una <b>aproximación</b>.
                    </div>
                </div>
            </div>
        </div>
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Sube una imagen de una hoja",
            height=500
        )

        result_output = gr.HTML(
            value=placeholder_card(),
            label="Diagnóstico"
        )

    with gr.Row():
        btn_predict = gr.Button("Clasificar hoja", variant="primary")
        btn_clear = gr.Button("Limpiar")

    btn_predict.click(
        fn=predict,
        inputs=image_input,
        outputs=result_output
    )

    btn_clear.click(
        fn=clear_interface,
        inputs=[],
        outputs=[image_input, result_output]
    )

    gr.HTML(
        """
        <div class="recommend-box">
            <h3>✅ Recomendaciones para mejores resultados</h3>
            <ul>
                <li>Usa una imagen clara de la hoja.</li>
                <li>Evita fondos demasiado cargados.</li>
                <li>Procura buena iluminación.</li>
                <li>En lo posible, enfoca una hoja principal.</li>
                <li>Si la planta no pertenece a los cultivos entrenados, interpreta el resultado como aproximado.</li>
            </ul>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)