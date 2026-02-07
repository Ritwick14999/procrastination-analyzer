
import matplotlib.pyplot as plt
import pandas as pd

def plot_hour_distribution(df: pd.DataFrame):
    hours = df["ts"].dt.hour.value_counts().sort_index()
    fig = plt.figure()
    hours.plot(kind="bar")
    plt.xlabel("Hour of Day")
    plt.ylabel("Activity Count")
    plt.title("Activity by Hour")
    plt.tight_layout()
    return fig

def plot_daily_activity(df: pd.DataFrame):
    daily = df.groupby(df["ts"].dt.date).size()
    fig = plt.figure()
    daily.plot(kind="line", marker="o")
    plt.xlabel("Date")
    plt.ylabel("Events")
    plt.title("Daily Activity Trend")
    plt.tight_layout()
    return fig

def plot_bucket_pie(df: pd.DataFrame):
    def bucket(h):
        if 5 <= h < 12: return "morning"
        if 12 <= h < 17: return "afternoon"
        if 17 <= h < 22: return "evening"
        return "late_night"

    b = df["ts"].dt.hour.apply(bucket).value_counts()
    fig = plt.figure()
    b.plot(kind="pie", autopct="%1.0f%%")
    plt.ylabel("")
    plt.title("Time-of-Day Breakdown")
    plt.tight_layout()
    return fig
