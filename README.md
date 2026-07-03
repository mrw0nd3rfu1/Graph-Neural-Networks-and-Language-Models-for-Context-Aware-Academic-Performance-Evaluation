# Explainable GNN-Based Academic Performance Recommendation System

A Graph Neural Network (GNN) recommendation system that suggests personalized, evidence-based study strategies to students based on peer similarity, evaluates recommendation quality using **RAGAS** (LLM-judged) and ranking metrics, and provides full **Explainable AI (XAI)** analysis of the trained model's decisions.

This repository contains two notebooks:

| Notebook | Purpose |
|---|---|
| `model.ipynb` | Builds and trains the GNN recommender, runs a 50-seed model selection sweep, and evaluates the best model with both GNN-only ranking metrics (Precision@k, Recall@k, NDCG@k) and RAGAS metrics (faithfulness, answer relevancy, semantic similarity, context recall, answer correctness). |
| `xai_gnn.ipynb` | Loads the best trained GNN and produces a complete Explainable AI report: GAT attention-weight visualization, GNNExplainer, feature attribution, integrated gradients, optional SHAP values, and qualitative/embedding-based analysis for any student node. |

## How It Works

1. **Data & Graph Construction** — Survey responses (academic performance, stress, confidence, and free-text challenges) are encoded with `sentence-transformers` (`all-MiniLM-L6-v2`). A similarity graph is built by connecting students whose embeddings exceed a cosine-similarity threshold.
2. **GNN Model** — A 2-layer Graph Attention Network (`GATConv`, PyTorch Geometric) is trained to reconstruct student feature vectors from their neighborhood, learning representations that capture peer similarity in academic profile.
3. **Recommendation** — For a new student query, the system embeds the query, scores it against learned node representations (blending semantic similarity with academic-profile similarity), and selects a diverse top-k set of peer responses.
4. **LLM Summarization** — A local LLM served via **Ollama** (e.g. `llama3.2`) turns the top-k peer responses into a structured, actionable study plan (immediate actions, weekly strategies, long-term habits).
5. **Evaluation**
   - *GNN-only:* Precision@k, Recall@k, NDCG@k against a ground-truth JSON set.
   - *RAGAS (LLM-based):* faithfulness, answer relevancy, semantic similarity, context recall, and answer correctness, with a manual TF-IDF-based fallback if RAGAS fails.
   - A 50-seed sweep trains the GNN from scratch under different random seeds and keeps the checkpoint with the best combined GNN score.
6. **Explainability** — The XAI notebook reloads the best checkpoint and explains individual recommendations via GAT attention weights, GNNExplainer, gradient/SHAP-based feature attribution, and human-readable qualitative pattern analysis, producing plots and a text report per student.

## Repository Structure

```
.
├── model.ipynb       # Training, seed sweep, and RAGAS/GNN evaluation
├── xai_gnn.ipynb    # Explainable AI analysis of the trained GNN
├── Dataset.csv          # Student survey dataset (not included — see Data below)
├── rag_eval_pairs.json   # Ground-truth query/answer pairs for evaluation (not included)
└── best_gnn_model.pt     # Saved best GNN checkpoint (produced by testing7.ipynb)
```

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally, with a model pulled (default: `llama3.2:latest`)

Install Python dependencies:

```bash
pip install pandas numpy scikit-learn torch torch_geometric \
            sentence-transformers datasets ragas requests \
            matplotlib seaborn networkx
```

Optional (for SHAP-based attribution in the XAI notebook):

```bash
pip install shap
```

> `torch_geometric` installation is hardware/CUDA-version specific — follow the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your setup.

## Data

The notebooks expect:
- `Dataset3.csv` — student survey responses, including columns such as *"What was your performance in previous academic levels?"*, *"How often do you feel stressed about academics?"*, and *"How confident are you in your academic abilities?"*, plus free-text challenge fields.
- `rag_eval_pairs.json` — a list of `{"user_input": {...}, "ground_truth": "..."}` objects used as the evaluation/ground-truth set.

These files are not included in this repository; supply your own dataset in the same format, or reach out for details.

## Usage

1. **Start Ollama** and pull a model:
   ```bash
   ollama serve
   ollama pull llama3.2:latest
   ```
2. **Train and evaluate the GNN** — run `testing7.ipynb` end-to-end. This performs the 50-seed sweep, saves the best model to `best_gnn_model.pt`, and writes `best_gnn_sweep_results.json` and `best_gnn_ragas_results.json`.
3. **Run the XAI analysis** — run `testing7xai.ipynb`, pointing `GNNXAIAnalyzer` at `Dataset3.csv` and the saved `best_gnn_model.pt`. Set `target_student` to the row index of the student you want to explain. Results (plots + text report) are saved to `xai_results/`.

## Citation

This code accompanies the following peer-reviewed paper. If you use this repository in your research, please cite:

> A. Pandey et al., *"[Paper Title]"*, IEEE International Conference on Artificial Intelligence (IEEE-CAI), 2026. Available: https://ieeexplore.ieee.org/abstract/document/11536196

```bibtex
@inproceedings{pandey2026gnn,
  title     = {Graph Neural Networks and Language Models for Academic Performance Evaluation},
  author    = {Pandey, Abhinav and others},
  booktitle = {IEEE International Conference on Artificial Intelligence (IEEE-CAI)},
  year      = {2026},
  url       = {https://ieeexplore.ieee.org/abstract/document/11536196}
}
```
