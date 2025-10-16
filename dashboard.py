import streamlit as st
import json
import os

port = int(os.environ.get("PORT", 8501))

st.set_page_config(page_title="NeuralRoulette Dashboard", layout="wide")
st.title("🎯 NeuralRoulette Dashboard")

try:
    with open("game_history.json", "r") as f:
        history = json.load(f)
except:
    history = {"results": [], "wins": 0, "balance": 10.0, "roi": 0.0, "last_predictions": []}

st.metric("Total Spins", len(history["results"]))
st.metric("Acertos", history["wins"])
st.metric("Win Rate", f'{history["wins"] / len(history["results"]):.2%}' if history["results"] else "0.00%")
st.metric("Saldo Atual", f'${history["balance"]:.2f}')
st.metric("ROI", f'{history["roi"]:.2%}')

st.write("🔮 Últimas predições:")
st.write(history["last_predictions"][-5:])
