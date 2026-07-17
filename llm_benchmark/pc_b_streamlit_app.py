import os

import pandas as pd
import requests
import streamlit as st

PC_A_URL = os.getenv("PC_A_URL", "http://100.x.x.x:8000")
MODELS = ["llama3.1", "mistral", "phi3", "qwen2"]

st.set_page_config(page_title="LLM Benchmark", layout="wide")
st.title("LLM Benchmark: Llama 3.1 vs Mistral vs Phi-3 Mini vs Qwen2")

prompt = st.text_area(
    "Prompt", "Explain how a robot can navigate an expo hall safely."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Chat with one model")
    single_model = st.selectbox("Model", MODELS)
    if st.button("Send"):
        with st.spinner(f"Querying {single_model}..."):
            resp = requests.post(
                f"{PC_A_URL}/benchmark",
                json={"model": single_model, "prompt": prompt},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
        st.write(data["response"])
        st.json({k: v for k, v in data.items() if k != "response"})

with col2:
    st.subheader("Benchmark all 4 models")
    if st.button("Run benchmark"):
        results = []
        progress = st.progress(0.0)
        for i, model in enumerate(MODELS):
            with st.spinner(f"Running {model}..."):
                resp = requests.post(
                    f"{PC_A_URL}/benchmark",
                    json={"model": model, "prompt": prompt},
                    timeout=300,
                )
                resp.raise_for_status()
                results.append(resp.json())
            progress.progress((i + 1) / len(MODELS))

        df = pd.DataFrame(results)[
            [
                "model",
                "tokens_per_second",
                "eval_count",
                "total_duration_s",
                "load_duration_s",
                "wall_time_s",
            ]
        ]
        st.subheader("Benchmark Results")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("model")["tokens_per_second"])

        with st.expander("Full model responses"):
            for r in results:
                st.markdown(f"**{r['model']}**")
                st.write(r["response"])
                st.divider()

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV", csv, "llm_benchmark_results.csv", "text/csv"
        )
