# Etapa 5: Tablero de datos

Tablero en Dash para un **anfitrión nuevo en Las Vegas**: estima un precio por noche, indica si su anuncio sería recomendable (red neuronal de la Etapa 4b) y muestra 3 gráficas del mercado.

## Archivos
- `app.py`: código del tablero.
- `assets/estilo.css`: estilos (colores, tamaños).
- `modelo_recomendable.keras`: red neuronal de clasificación (Etapa 4b), ya entrenada.
- `config_modelo.json`: columnas que usa el modelo y cuáles llevan logaritmo.
- `listings_clark_limpio.csv`: datos limpios (Etapa 2).

## Cómo correrlo
```
pip install dash pandas numpy plotly tensorflow
python app.py
```
Luego abrir http://localhost:8050

## Pendiente
El precio sugerido es provisional (mediana de anuncios parecidos) hasta integrar el modelo de regresión.
