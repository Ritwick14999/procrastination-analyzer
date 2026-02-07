# Procrastination Pattern Analyzer

A small data-focused project that analyzes activity timestamps to spot procrastination patterns, estimate short-term risk, and suggest practical improvements.

The idea behind this project is : timing and frequency of work often reveal useful signals about productivity habits. This project explores how far those signals can go when turned into a lightweight analytics tool.


#Features

- Detects procrastination patterns from timestamp logs  
- Estimates next-day procrastination risk  
- Generates an avoidance/perfectionism score  
- Visualizes activity trends  
- Retrieves relevant suggestions using a simple RAG setup  
- Exports a markdown report  
- Interactive Streamlit dashboard  


# How It Works

1. Timestamp data is provided through CSV upload or manual input  
2. The analysis module extracts behavioral signals such as:
   - inactivity gaps  
   - burst work sessions  
   - late-night activity  
   - weekend activity  
3. A rule-based model uses these signals to:
   - classify a procrastination pattern  
   - compute an avoidance/perfectionism score  
   - estimate next-day risk  
4. A TF-IDF retrieval layer selects relevant suggestion snippets  
5. Results are displayed in a Streamlit dashboard with charts and summaries  


#📂 Project Structure
procrastination_analyzer/
├── analysis/ # pattern detection + scoring logic
├── rag/ # snippet retrieval (TF-IDF)
├── report/ # markdown report generation
├── ui/ # streamlit app + visualizations
├── data/ # sample dataset
├── output/ # generated reports
└── tests/ # basic pytest checks


# Tech Stack
- Python  
- Pandas / NumPy  
- Scikit-learn (TF-IDF retrieval)  
- Streamlit  
- Matplotlib  
- PyTest  


#▶️ Running the App

### Locally

```bash
pip install -r requirements.txt
streamlit run procrastination_analyzer/ui/app.py
```
#On Google Colab

The app can also be run on Colab and exposed publicly using a cloudflared tunnel.

⚠️ Note:
This is an experimental, rule-based system and not a psychological diagnostic tool
Scores are heuristic and meant for exploration
Suggestion snippets were generated with AI assistance and curated for this project

# Possible Improvements:
ML-based prediction instead of heuristics
User history tracking for better personalization
More evaluation on real-world data
Additional input sources (task logs, calendars, etc.)

⭐ Purpose
Built as a learning project around:
behavioral data analysis
lightweight RAG systems
building end-to-end data apps

