# Topic Modeling and Trend Analysis of Research Papers

A Python web application that uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze research papers. It helps students and researchers discover topics, track research trends, find similar papers, and spot possible research gaps.

## Problem Statement
Thousands of research papers are published every year, and reading them manually takes a lot of time. This project automates the process using NLP, topic modeling, and semantic similarity.

## Features
- **Home Dashboard:** total papers, topics discovered, years covered, and papers per year.
- **Topic Explorer:** view automatically discovered topics, their keywords, and the papers in each topic.
- **Research Trends:** interactive charts showing which topics are growing, declining, or stable.
- **Similar Paper Search:** enter a research idea or abstract and get the most similar existing papers with similarity scores, plus a short research-gap hint.
- **Paper Search and Details:** search papers by keyword and year and view their metadata.

## How It Works
1. **Collect data:** arXiv research papers dataset (title, abstract, category, year). About 8,000 Computer Science papers are used to keep processing light.
2. **Clean text:** remove symbols and stop words, tokenize, and lemmatize with spaCy.
3. **Embeddings:** convert each paper into a vector using the Sentence-Transformers model `all-MiniLM-L6-v2`, so the system understands meaning, not just keywords.
4. **Topic modeling:** group similar papers into topics with BERTopic (UMAP + HDBSCAN + c-TF-IDF).
5. **Trend analysis:** calculate each topic's share of papers per year.
6. **Similar paper search:** compare a user's idea with all paper embeddings using cosine similarity.

## Tech Stack
| Area | Tools |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Text preprocessing | spaCy |
| Embeddings | Sentence-Transformers |
| Topic modeling | BERTopic, UMAP, HDBSCAN |
| Similarity | Scikit-learn (cosine similarity) |
| Visualization | Plotly |
| Web app | Gradio (Colab), Streamlit (`app1.py`) |
| Dataset | arXiv dataset (Kaggle / Hugging Face copy) |

## Project Structure
```
research-topic-modeling/
├── app1.py                              # Streamlit app
├── Research_Topic_Modeling_Colab.ipynb  # Full pipeline + Gradio app for Google Colab
├── requirements.txt
└── README.md
```

## How to Run

### Option 1: Google Colab (recommended)
1. Open `Research_Topic_Modeling_Colab.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**.
4. Open the public link printed by the last cell to use the web app.

### Option 2: Run locally with Streamlit (Windows)
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app1.py
```

## Dataset
- Kaggle (official arXiv dataset): https://www.kaggle.com/datasets/Cornell-University/arxiv
- Hugging Face copy (used by default in the notebook): https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021

The dataset is large and is **not included** in this repository. The notebook downloads and samples about 8,000 Computer Science papers automatically. To use Kaggle directly, set `SOURCE = "kaggle"` in the notebook and upload your own `kaggle.json`.

## Example
Input idea: *"Using transformer models to detect fake news on social media"*
Output: the most similar existing papers, their topics, similarity scores, and a note on how explored the idea already is.

## Limitations and Future Work
- Uses a sample of about 8,000 papers, not the full 1.7 million.
- Year is taken from arXiv submission or update date.
- Future work: use the full dataset, add citation data, and support more research fields.

## Author
**Your Name**, B.Tech (AI), Your College Name
GitHub: https://github.com/vinay0varma

## License
This project is for educational purposes.
