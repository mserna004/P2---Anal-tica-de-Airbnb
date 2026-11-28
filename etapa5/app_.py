# ============================================================
# Tablero Airbnb Las Vegas
# Usuario: anfitrión nuevo que quiere saber cuánto cobrar.
# Los estilos (colores, tamaños) están en assets/estilo.css
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from dash import Dash, html, dcc, Input, Output, State

CARPETA = os.path.dirname(os.path.abspath(__file__))
VERDE, GRIS = "#0E6E62", "#C5CDC9"


# ------------------------------------------------------------
# 1. Textos del formulario (valor en los datos → texto en pantalla)
# ------------------------------------------------------------
ZONAS = {"Unincorporated Areas": "Strip y alrededores",
         "City of Las Vegas": "Ciudad de Las Vegas (Downtown)",
         "City of Henderson": "Henderson",
         "City of North Las Vegas": "North Las Vegas",
         "City of Mesquite": "Mesquite",
         "Other": "Otras (Boulder City, Nellis)"}
ALOJAMIENTOS = {"Entire home/apt": "Casa o apartamento completo",
                "Private room": "Habitación privada",
                "Shared room": "Habitación compartida",
                "Hotel room": "Habitación de hotel"}
PROPIEDADES = {"House": "Casa", "Apartment": "Apartamento", "Condo": "Condominio",
               "Hotel/Resort": "Hotel / Resort", "Guesthouse": "Casa de huéspedes", "Other": "Otro"}
COMODIDADES = {"has_pool": "Piscina", "has_hot_tub": "Jacuzzi", "has_parking": "Parqueadero gratis",
               "has_kitchen": "Cocina", "has_washer": "Lavadora", "has_workspace": "Espacio de trabajo",
               "has_self_checkin": "Self check-in", "has_gym": "Gimnasio", "has_pets": "Acepta mascotas"}


# ------------------------------------------------------------
# 2. Datos
# ------------------------------------------------------------
df = pd.read_csv(os.path.join(CARPETA, "listings_clark_limpio.csv"))

# La zona y el tipo vienen como columnas 1/0. Aquí se vuelven a juntar en una sola columna.
# Si todas están en 0, es la categoría que se quitó al codificar (drop_first).
def juntar_columnas(prefijo, categoria_quitada):
    cols = [c for c in df.columns if c.startswith(prefijo)]
    valor = df[cols].idxmax(axis=1).str.replace(prefijo, "", regex=False)
    return valor.where(df[cols].sum(axis=1) == 1, categoria_quitada)

df["zona"] = juntar_columnas("neighbourhood_cleansed_", "City of Henderson")
df["alojamiento"] = juntar_columnas("room_type_", "Entire home/apt")


# ------------------------------------------------------------
# 3. Modelo de clasificación (red neuronal de la Etapa 4b)
# ------------------------------------------------------------
MODELO = tf.keras.models.load_model(os.path.join(CARPETA, "modelo_recomendable.keras"))
CONFIG = json.load(open(os.path.join(CARPETA, "config_modelo.json")))  # columnas y cuáles llevan log

# Lo que el formulario no pregunta se llena con valores típicos de los datos
VALORES_FIJOS = {"hosts_time_as_host_years": 5, "host_is_superhost": 0, "host_has_profile_pic": 1,
                 "host_identity_verified": 1, "calculated_host_listings_count": 1, "host_profesional": 0,
                 "minimum_nights": 2, "maximum_nights": 365, "availability_30": 19,
                 "availability_365": 264, "bathrooms_shared": 0, "is_exempt": 0, "has_license": 0,
                 "licencia_desconocida": 1, "n_amenities": 45}
CAMAS_TIPICAS = df.groupby("accommodates")["beds"].median().to_dict()
CENTRO_ZONA = df.groupby("zona")[["latitude", "longitude"]].median()


def km_al_strip(lat, lon):
    """Distancia en km al Strip (mismo punto que usó la Etapa 3)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat, lon, 36.1147, -115.1728])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def preparar_datos(zona, alojamiento, propiedad, huespedes, habitaciones, banos, comodidades, precio):
    """Convierte lo que llenó el usuario en una fila igual a las que vio la red al entrenar."""
    lat, lon = CENTRO_ZONA.loc[zona]
    fila = dict(VALORES_FIJOS)
    fila.update({"latitude": lat, "longitude": lon, "dist_strip_km": km_al_strip(lat, lon),
                 "accommodates": huespedes, "bedrooms": habitaciones, "bathrooms": banos,
                 "beds": CAMAS_TIPICAS.get(huespedes, 2),
                 "log_price": np.log(precio), "price_per_guest": precio / huespedes})
    for c in COMODIDADES:                                   # comodidades: 1 si la marcó
        fila[c] = int(c in comodidades)
    for c in CONFIG["columnas"]:                            # zona y tipos: 1 solo en la elegida
        if c.startswith(("room_type_", "property_type_", "neighbourhood_cleansed_")):
            fila[c] = int(c in (f"room_type_{alojamiento}", f"property_type_{propiedad}",
                                f"neighbourhood_cleansed_{zona}"))
    for c in CONFIG["log"]:                                 # mismo logaritmo que en la Etapa 4b
        fila[c] = np.log1p(fila[c])
    return pd.DataFrame([fila])[CONFIG["columnas"]].astype("float32")


def precio_sugerido(zona, alojamiento, huespedes):
    """PROVISIONAL (hasta tener la regresión): mediana de anuncios parecidos."""
    parecidos = df[(df.zona == zona) & (df.alojamiento == alojamiento) & (df.accommodates == huespedes)]
    if len(parecidos) < 20:  # si hay muy pocos, se compara solo por tipo y huéspedes
        parecidos = df[(df.alojamiento == alojamiento) & (df.accommodates == huespedes)]
    return parecidos["price"].median(), len(parecidos)


# ------------------------------------------------------------
# 4. Datos para las gráficas
# ------------------------------------------------------------
PRECIO_POR_ZONA = df.groupby(df["zona"].map(ZONAS))["price"].median().sort_values()

def diferencia_por_comodidad(col):
    """% de diferencia de precio con vs. sin la comodidad, entre anuncios con los mismos huéspedes."""
    med = df.groupby(["accommodates", col])["price"].median().unstack()
    n = df.groupby("accommodates")[col].agg(["sum", "count"])
    hay_datos = (n["sum"] >= 30) & (n["count"] - n["sum"] >= 30)
    return np.average((med[1] / med[0] - 1)[hay_datos], weights=n["count"][hay_datos]) * 100

DIFERENCIA_COMODIDAD = pd.Series({c: diferencia_por_comodidad(c) for c in COMODIDADES}).sort_values()

mapa = px.scatter_map(
    df, lat="latitude", lon="longitude", color="price", zoom=10, opacity=0.6,
    center={"lat": 36.13, "lon": -115.17}, map_style="carto-positron",
    color_continuous_scale=["#DCE5E1", "#8FB8AE", VERDE, "#063C35"], range_color=[50, 600],
    hover_data={"price": ":$,.0f", "latitude": False, "longitude": False}, labels={"price": "Precio"})
mapa.update_traces(marker_size=5)
mapa.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0),
                   coloraxis_colorbar=dict(title="USD/noche", tickprefix="$"))


def grafica_barras(valores, nombres, textos, colores, rango):
    fig = go.Figure(go.Bar(x=valores, y=nombres, orientation="h", marker_color=colores,
                           text=textos, textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=300, margin=dict(l=10, r=30, t=10, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", showlegend=False)
    fig.update_xaxes(visible=False, range=rango)
    return fig


# ------------------------------------------------------------
# 5. Diseño del tablero
# ------------------------------------------------------------
def lista(opciones):
    return [{"label": texto, "value": valor} for valor, texto in opciones.items()]

def campo(etiqueta, control):
    return html.Div([html.Label(etiqueta), control], className="campo")

def tarjeta_resultado(titulo, id_valor, id_nota):
    return html.Div([html.Div(titulo, className="antetitulo"),
                     html.Div("—", id=id_valor, className="numero"),
                     html.Div(id=id_nota, className="nota")], className="tarjeta resultado")

def titulo_grafica(numero, titulo, nota):
    return html.Div([html.Div(f"Gráfica {numero}", className="antetitulo"),
                     html.H4(titulo), html.Div(nota, className="nota")])


app = Dash(__name__)
app.title = "Airbnb Las Vegas"

app.layout = html.Div(className="pagina", children=[
    html.Div("Airbnb · Clark County, Nevada", className="antetitulo"),
    html.H1("¿Cuánto cobrar por tu Airbnb en Las Vegas?"),
    html.P("Llena los datos de tu propiedad y da clic en «Calcular». Verás un precio sugerido por noche "
           "y si tu anuncio sería recomendable. Más abajo puedes comparar con el mercado.",
           className="intro"),

    html.Div(className="fila", children=[
        # Paso 1: formulario
        html.Div(className="tarjeta columna", children=[
            html.Div("Paso 1", className="antetitulo"), html.H3("Tu propiedad"),
            campo("Zona", dcc.Dropdown(lista(ZONAS), "Unincorporated Areas", id="zona", clearable=False)),
            campo("Tipo de alojamiento", dcc.Dropdown(lista(ALOJAMIENTOS), "Entire home/apt",
                                                      id="alojamiento", clearable=False)),
            campo("Tipo de propiedad", dcc.Dropdown(lista(PROPIEDADES), "House",
                                                    id="propiedad", clearable=False)),
            html.Div(className="fila-pequena", children=[
                campo("Huéspedes", dcc.Input(id="huespedes", type="number", value=4, min=1, max=16)),
                campo("Habitaciones", dcc.Input(id="habitaciones", type="number", value=2, min=0, max=16)),
                campo("Baños", dcc.Input(id="banos", type="number", value=1, min=0, max=15, step=0.5)),
            ]),
            campo("¿A qué precio piensas publicarlo? (USD por noche, opcional)",
                  dcc.Input(id="precio_usuario", type="number", min=20, max=2000,
                            placeholder="Si lo dejas vacío, usamos el precio sugerido")),
            campo("Comodidades", dcc.Checklist(lista(COMODIDADES), ["has_kitchen", "has_parking"],
                                               id="comodidades", className="casillas")),
            html.Button("Calcular", id="calcular", n_clicks=0, className="boton"),
        ]),

        # Paso 2: resultados
        html.Div(className="columna", children=[
            html.Div("Paso 2", className="antetitulo"), html.H3("Resultado"),
            html.Div(className="fila-pequena", children=[
                tarjeta_resultado("Precio sugerido por noche", "precio", "nota_precio"),
                tarjeta_resultado("¿Anuncio recomendable?", "recomendable", "nota_recomendable"),
            ]),
            html.Div("Llena el formulario y da clic en «Calcular».", id="consejo", className="tarjeta nota"),
        ]),
    ]),

    # Paso 3: gráficas del mercado
    html.Div(className="tarjeta", children=[
        html.Div("Paso 3", className="antetitulo"), html.H3("El mercado en Las Vegas"),
        html.P(f"Basado en {len(df):,} anuncios de Inside Airbnb (septiembre de 2026).".replace(",", "."),
               className="nota"),
        titulo_grafica(1, "¿Dónde están los anuncios más caros?",
                       "Cada punto es un anuncio. Entre más oscuro, más alto el precio por noche."),
        dcc.Graph(figure=mapa, config={"displayModeBar": False}),
        html.Div(className="fila", children=[
            html.Div(className="columna", children=[
                titulo_grafica(2, "Precio típico por zona",
                               "Precio mediano por noche. En verde, la zona que elegiste."),
                dcc.Graph(id="grafica_zonas", config={"displayModeBar": False})]),
            html.Div(className="columna", children=[
                titulo_grafica(3, "¿Qué comodidades suben el precio?",
                               "Diferencia de precio frente a anuncios para el mismo número de huéspedes. "
                               "En verde, las que marcaste."),
                dcc.Graph(id="grafica_comodidades", config={"displayModeBar": False})]),
        ]),
    ]),
])


# ------------------------------------------------------------
# 6. Interacción
# ------------------------------------------------------------
@app.callback(
    Output("precio", "children"), Output("nota_precio", "children"),
    Output("recomendable", "children"), Output("nota_recomendable", "children"),
    Output("consejo", "children"),
    Input("calcular", "n_clicks"),
    State("zona", "value"), State("alojamiento", "value"), State("propiedad", "value"),
    State("huespedes", "value"), State("habitaciones", "value"), State("banos", "value"),
    State("comodidades", "value"), State("precio_usuario", "value"),
    prevent_initial_call=True,
)
def calcular(_, zona, alojamiento, propiedad, huespedes, habitaciones, banos, comodidades, precio_usuario):
    huespedes, habitaciones, banos = int(huespedes or 1), habitaciones or 0, banos or 1
    datos = [zona, alojamiento, propiedad, huespedes, habitaciones, banos]

    # Precio sugerido (provisional) y precio con el que se evalúa el anuncio
    sugerido, n_parecidos = precio_sugerido(zona, alojamiento, huespedes)
    precio = precio_usuario or sugerido

    # Probabilidad de ser recomendable
    prob = MODELO.predict(preparar_datos(*datos, comodidades, precio), verbose=0)[0, 0]

    # Consejo: ¿cuánto subiría la probabilidad si agrega cada comodidad que le falta?
    mejoras = []
    for c in COMODIDADES:
        if c not in comodidades:
            nueva = MODELO.predict(preparar_datos(*datos, comodidades + [c], precio), verbose=0)[0, 0]
            if nueva - prob >= 0.02:
                mejoras.append((nueva - prob, COMODIDADES[c].lower()))
    mejoras = sorted(mejoras, reverse=True)[:2]
    if mejoras:
        consejo = "Lo que más subiría tu probabilidad: " + "; ".join(
            f"agregar {nombre} (+{d * 100:.0f} puntos)" for d, nombre in mejoras) + "."
    else:
        consejo = "Tu anuncio ya tiene las comodidades que más ayudan."

    return (f"${sugerido:,.0f}", f"Mediana de {n_parecidos} anuncios parecidos",
            "Sí" if prob >= 0.5 else "No",
            f"{prob * 100:.0f} % de probabilidad, cobrando ${precio:,.0f} por noche",
            consejo)


@app.callback(Output("grafica_zonas", "figure"), Input("zona", "value"))
def actualizar_zonas(zona):
    colores = [VERDE if z == ZONAS[zona] else GRIS for z in PRECIO_POR_ZONA.index]
    textos = [f"${v:,.0f}" for v in PRECIO_POR_ZONA]
    return grafica_barras(PRECIO_POR_ZONA, PRECIO_POR_ZONA.index, textos, colores,
                          [0, PRECIO_POR_ZONA.max() * 1.2])


@app.callback(Output("grafica_comodidades", "figure"), Input("comodidades", "value"))
def actualizar_comodidades(marcadas):
    d = DIFERENCIA_COMODIDAD
    colores = [VERDE if c in marcadas else GRIS for c in d.index]
    textos = [f"{v:+.0f} %" for v in d]
    fig = grafica_barras(d, [COMODIDADES[c] for c in d.index], textos, colores,
                         [d.min() - 15, d.max() + 15])
    fig.add_vline(x=0, line_color="#9AA5A1", line_width=1)
    return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
