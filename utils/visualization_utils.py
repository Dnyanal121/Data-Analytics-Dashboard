import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from .data_utils import get_sample

def generate_plot(df, x, y, chart_type, sample_threshold=500000):
    df_sample = get_sample(df, sample_threshold)
    fig_html = None

    try:
        if chart_type == "hist":
            fig = px.histogram(df_sample, x=x)
        elif chart_type == "scatter":
            fig = px.scatter(df_sample, x=x, y=y)
        elif chart_type == "box":
            fig = px.box(df_sample, x=x, y=y)
        elif chart_type == "heatmap":
            fig = px.imshow(df_sample.corr(), text_auto=True, aspect="auto")
        else:
            fig = px.histogram(df_sample, x=x)
        fig_html = fig.to_html(full_html=False)
    except Exception as e:
        print("Plot error:", e)
    return fig_html
