from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

st.set_page_config(
    page_title="House Prices — Regressão e Clusterização",
    page_icon="🏠",
    layout="wide",
)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    def col(name: str) -> pd.Series:
        if name in data.columns:
            return pd.to_numeric(data[name], errors="coerce").fillna(0)
        return pd.Series(0, index=data.index, dtype=float)

    data["TotalSF"] = col("TotalBsmtSF") + col("1stFlrSF") + col("2ndFlrSF")
    data["TotalBathrooms"] = (
        col("FullBath")
        + 0.5 * col("HalfBath")
        + col("BsmtFullBath")
        + 0.5 * col("BsmtHalfBath")
    )
    data["TotalPorchSF"] = (
        col("OpenPorchSF")
        + col("EnclosedPorch")
        + col("3SsnPorch")
        + col("ScreenPorch")
        + col("WoodDeckSF")
    )
    data["HouseAge"] = col("YrSold") - col("YearBuilt")
    data["RemodAge"] = col("YrSold") - col("YearRemodAdd")
    data["HasGarage"] = (col("GarageArea") > 0).astype(int)
    data["HasBasement"] = (col("TotalBsmtSF") > 0).astype(int)
    data["HasFireplace"] = (col("Fireplaces") > 0).astype(int)
    data["HasPool"] = (col("PoolArea") > 0).astype(int)
    return data


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_model(model_name: str, params: dict) -> object:
    if model_name == "Ridge":
        return Ridge(alpha=params["alpha"])

    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return GradientBoostingRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        loss=params["loss"],
        random_state=RANDOM_STATE,
    )


def evaluate_model(
    train_df: pd.DataFrame,
    model_name: str,
    params: dict,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    data = create_features(train_df)

    if "SalePrice" not in data.columns:
        raise ValueError("O arquivo de treino precisa conter a coluna SalePrice.")

    drop_cols = [c for c in ["SalePrice", "Id"] if c in data.columns]
    X = data.drop(columns=drop_cols)
    y = np.log1p(data["SalePrice"])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X)),
            ("model", build_model(model_name, params)),
        ]
    )
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_valid)
    real = np.expm1(y_valid)
    pred = np.clip(np.expm1(pred_log), 0, None)

    metrics = pd.DataFrame(
        {
            "Métrica": ["MAE", "RMSE", "RMSLE", "R²"],
            "Valor": [
                mean_absolute_error(real, pred),
                mean_squared_error(real, pred) ** 0.5,
                mean_squared_log_error(real, pred) ** 0.5,
                r2_score(real, pred),
            ],
        }
    )

    predictions = pd.DataFrame(
        {
            "Preço real": real.to_numpy(),
            "Preço previsto": pred,
            "Resíduo": real.to_numpy() - pred,
        }
    )

    # Reajusta com todo o conjunto para uso em submissão
    pipeline.fit(X, y)
    return pipeline, metrics, predictions


def cluster_houses(
    train_df: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    data = create_features(train_df)

    preferred = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "GarageCars",
        "YearBuilt",
        "TotalBathrooms",
        "TotalSF",
    ]
    features = [c for c in preferred if c in data.columns]

    if len(features) < 2:
        features = data.select_dtypes(include=["number"]).columns.tolist()
        features = [
            c for c in features if c not in ["Id", "SalePrice"]
        ][:7]

    if len(features) < 2:
        raise ValueError(
            "Não há variáveis numéricas suficientes para a clusterização."
        )

    cluster_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    scaled = cluster_pipe.fit_transform(data[features])

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    labels = model.fit_predict(scaled)

    clustered = data.copy()
    clustered["Cluster"] = labels

    profile_columns = features.copy()
    if "SalePrice" in clustered.columns:
        profile_columns.append("SalePrice")

    profile = (
        clustered.groupby("Cluster")[profile_columns]
        .mean()
        .round(2)
    )
    score = silhouette_score(scaled, labels)
    return clustered, profile, score


st.title("🏠 House Prices")
st.caption("Regressão supervisionada, hiperparâmetros e clusterização com K-Means.")

with st.sidebar:
    st.header("Dados")
    train_file = st.file_uploader(
        "Envie o train.csv",
        type=["csv"],
    )
    test_file = st.file_uploader(
        "Envie o test.csv (opcional)",
        type=["csv"],
    )
    sample_file = st.file_uploader(
        "Envie o sample_submission.csv (opcional)",
        type=["csv"],
    )

if train_file is None:
    st.info(
        "Envie o arquivo train.csv do Kaggle na barra lateral para iniciar."
    )
    st.stop()

train_df = pd.read_csv(train_file)

with st.sidebar:
    st.header("Modelo de regressão")
    model_name = st.selectbox(
        "Algoritmo",
        ["Gradient Boosting", "Random Forest", "Ridge"],
    )

    params: dict = {}
    if model_name == "Ridge":
        params["alpha"] = st.slider(
            "Alpha",
            0.1,
            50.0,
            10.0,
            0.1,
        )
    elif model_name == "Random Forest":
        params["n_estimators"] = st.slider(
            "Número de árvores",
            50,
            500,
            200,
            25,
        )
        params["max_depth"] = st.slider(
            "Profundidade máxima",
            3,
            30,
            18,
        )
        params["max_features"] = st.selectbox(
            "Máximo de variáveis por divisão",
            ["sqrt", "log2", 1.0],
        )
    else:
        params["n_estimators"] = st.slider(
            "Número de estimadores",
            50,
            500,
            200,
            25,
        )
        params["learning_rate"] = st.slider(
            "Taxa de aprendizado",
            0.01,
            0.20,
            0.05,
            0.01,
        )
        params["max_depth"] = st.slider(
            "Profundidade das árvores",
            1,
            6,
            3,
        )
        params["loss"] = st.selectbox(
            "Função de perda",
            ["huber", "squared_error"],
        )

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Visão geral",
        "Regressão",
        "Clusterização",
        "Submissão",
    ]
)

with tab1:
    st.subheader("Resumo do conjunto de dados")
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", f"{train_df.shape[0]:,}")
    c2.metric("Colunas", train_df.shape[1])
    c3.metric(
        "Valores ausentes",
        f"{int(train_df.isna().sum().sum()):,}",
    )

    st.dataframe(train_df.head(), use_container_width=True)

    if "SalePrice" in train_df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(train_df["SalePrice"].dropna(), bins=35)
        ax.set_title("Distribuição de SalePrice")
        ax.set_xlabel("Preço")
        ax.set_ylabel("Frequência")
        st.pyplot(fig)

with tab2:
    try:
        pipeline, metrics, predictions = evaluate_model(
            train_df,
            model_name,
            params,
        )

        st.subheader(f"Resultados — {model_name}")
        cols = st.columns(4)
        values = dict(zip(metrics["Métrica"], metrics["Valor"]))
        cols[0].metric("MAE", f"${values['MAE']:,.2f}")
        cols[1].metric("RMSE", f"${values['RMSE']:,.2f}")
        cols[2].metric("RMSLE", f"{values['RMSLE']:.4f}")
        cols[3].metric("R²", f"{values['R²']:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            predictions["Preço real"],
            predictions["Preço previsto"],
            alpha=0.5,
        )
        limit = max(
            predictions["Preço real"].max(),
            predictions["Preço previsto"].max(),
        )
        ax.plot([0, limit], [0, limit], linestyle="--")
        ax.set_xlabel("Preço real")
        ax.set_ylabel("Preço previsto")
        ax.set_title("Real × previsto")
        st.pyplot(fig)

        st.markdown(
            "**Hiperparâmetros atuais:** "
            + ", ".join(f"`{k}={v}`" for k, v in params.items())
        )
    except Exception as exc:
        st.error(f"Não foi possível treinar o modelo: {exc}")
        pipeline = None

with tab3:
    st.subheader("Agrupamento de imóveis semelhantes")
    k = st.slider("Número de clusters (k)", 2, 8, 4)

    try:
        clustered, profile, score = cluster_houses(train_df, k)
        st.metric("Coeficiente de silhueta", f"{score:.4f}")
        st.dataframe(profile, use_container_width=True)

        numeric_cols = clustered.select_dtypes(include=["number"]).columns
        default_x = "GrLivArea" if "GrLivArea" in numeric_cols else numeric_cols[0]
        default_y = "SalePrice" if "SalePrice" in numeric_cols else numeric_cols[1]

        c1, c2 = st.columns(2)
        x_col = c1.selectbox(
            "Eixo X",
            numeric_cols,
            index=list(numeric_cols).index(default_x),
        )
        y_col = c2.selectbox(
            "Eixo Y",
            numeric_cols,
            index=list(numeric_cols).index(default_y),
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            clustered[x_col],
            clustered[y_col],
            c=clustered["Cluster"],
            alpha=0.65,
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Visualização dos clusters")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Não foi possível executar o K-Means: {exc}")

with tab4:
    st.subheader("Gerar arquivo de submissão")

    if test_file is None:
        st.info("Envie o test.csv para gerar previsões.")
    elif pipeline is None:
        st.warning("O modelo precisa ser treinado antes da submissão.")
    else:
        test_df = pd.read_csv(test_file)
        test_eng = create_features(test_df)
        drop_cols = [c for c in ["Id"] if c in test_eng.columns]
        X_test = test_eng.drop(columns=drop_cols)

        test_pred = np.clip(
            np.expm1(pipeline.predict(X_test)),
            0,
            None,
        )

        if sample_file is not None:
            submission = pd.read_csv(sample_file)
        else:
            submission = pd.DataFrame()
            if "Id" in test_df.columns:
                submission["Id"] = test_df["Id"]

        submission["SalePrice"] = test_pred

        st.dataframe(submission.head(), use_container_width=True)

        csv_bytes = submission.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar submission_house_prices.csv",
            data=csv_bytes,
            file_name="submission_house_prices.csv",
            mime="text/csv",
        )
