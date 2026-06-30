---
title: Clasificador PlantVillage Equipo 07
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.45.0"
app_file: app.py
python_version: "3.10"
pinned: false
---

# Clasificador PlantVillage Equipo 07

Demo con Gradio y Hugging Face Spaces para clasificar enfermedades en hojas de plantas usando un modelo ConvNeXt Tiny entrenado con PlantVillage.

## Modelo usado

`Mardolco/vit-plantvillage-equipo07`

## Alcance del modelo

Puedes subir una imagen de una hoja de cualquier planta. Sin embargo, el modelo fue entrenado con clases específicas del dataset PlantVillage, por lo que sus predicciones son más confiables cuando la imagen pertenece a alguno de estos cultivos:

- Manzana
- Arándano
- Cereza
- Maíz
- Uva
- Naranja
- Durazno
- Pimiento
- Papa
- Frambuesa
- Soya
- Calabaza
- Fresa
- Tomate

Si se sube una hoja de otra planta, la aplicación igualmente devolverá una predicción, pero debe interpretarse como una aproximación.
