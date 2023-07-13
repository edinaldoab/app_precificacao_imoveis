pip install pandas
pip pip install -U scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Leitura do dataset
path = r'C:\Users\edina\OneDrive\Documentos\demo_apresentacao\dataset'
df = pd.read_parquet(path)

features = df.drop(['valor', 'bairro', 'id'], axis=1)
target = df['valor']

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=42)


@st.cache_data()
def train_model():
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    return rf_model


# Treinamento do modelo
rf_model = train_model()

# Centralizando o título
st.markdown("<h1 style='text-align: center;'>Previsão de valores de imóveis</h1>",
            unsafe_allow_html=True)

st.sidebar.title('Informações do imóvel')

with st.sidebar:
    st.markdown("## Localização:")
    zona_central = st.checkbox("Zona Central")
    zona_norte = st.checkbox("Zona Norte")
    zona_oeste = st.checkbox("Zona Oeste")
    zona_sul = st.checkbox("Zona Sul")

with st.sidebar:
    st.markdown("## Básicas:")
    andar = st.number_input("Andar (n°):", min_value=1)
    area_util = st.number_input("Área Útil (m²):", min_value=0.0)
    banheiros = st.number_input("Banheiros:", min_value=0)
    quartos = st.number_input("Quartos:", min_value=0)

with st.sidebar:
    st.markdown("## Adicionais")
    suites = st.number_input("Suítes:", min_value=0)
    vaga = st.checkbox("Vaga de Garagem")
    condominio = st.number_input("Valor do Condomínio (R$):", min_value=0.0)
    iptu = st.number_input("Valor do IPTU (R$):", min_value=0.0)
    academia = st.checkbox("Academia")
    animais_permitidos = st.checkbox("Permite animais")
    churrasqueira = st.checkbox("Churrasqueira")
    condominio_fechado = st.checkbox("Condomínio Fechado")
    elevador = st.checkbox("Elevador")
    piscina = st.checkbox("Piscina")
    playground = st.checkbox("Playground")
    portaria_24h = st.checkbox("Portaria 24h")
    portao_eletronico = st.checkbox("Portão Eletrônico")
    salao_de_festas = st.checkbox("Salão de Festas")

input_data = pd.DataFrame({
    'andar': [andar],
    'area_util': [area_util],
    'banheiros': [banheiros],
    'quartos': [quartos],
    'suites': [suites],
    'vaga': [vaga],
    'condominio': [condominio],
    'iptu': [iptu],
    'zona_central': [zona_central],
    'zona_norte': [zona_norte],
    'zona_oeste': [zona_oeste],
    'zona_sul': [zona_sul],
    'Academia': [academia],
    'animais_permitidos': [animais_permitidos],
    'Churrasqueira': [churrasqueira],
    'condominio_fechado': [condominio_fechado],
    'Elevador': [elevador],
    'Piscina': [piscina],
    'Playground': [playground],
    'portaria_24h': [portaria_24h],
    'portao_eletronico': [portao_eletronico],
    'salao_de_festas': [salao_de_festas]
})

prediction = rf_model.predict(input_data)
valor = f'{prediction[0]:,.2f}'\
                    .replace(",", " ")\
                    .replace(".", ",")\
                    .replace(" ", ".")

# Centralizando o texto da previsão
st.markdown("<h2 style='text-align: center;'>O imóvel deve ser precificado em <b>"
            + "R$ " + valor + "</b></h2>",
            unsafe_allow_html=True)
