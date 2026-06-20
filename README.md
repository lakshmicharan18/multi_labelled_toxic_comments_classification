🚫 Multi-Label Toxic Comment Classifier

A deep learning system that classifies social media comments into 6 toxic categories simultaneously, built with fine-tuned DistilBERT and deployed as a REST API.
📌 What This Project Does

Given any social media comment, the model predicts the probability of it belonging to 6 toxic categories — simultaneously in a single forward pass:

Input : "I will find you and hurt you badly"

Output:
  toxic         → 0.875  ✅ FLAGGED
  severe_toxic  → 0.466  clean
  obscene       → 0.452  clean
  threat        → 0.943  ✅ FLAGGED
  insult        → 0.423  clean
  identity_hate → 0.228  clean

  🏆 Results

Performance vs Baseline (TF-IDF + Logistic Regression)

LabelBaseline F1DistilBERT F1Gaintoxic0.78280.8574↑ 0.075severe_toxic0.48450.5330↑ 0.049obscene0.74010.8408↑ 0.101threat0.07550.5397↑ 0.464insult0.55400.7804↑ 0.226identity_hate0.09440.6101↑ 0.516Mean0.45520.6936↑ 0.238

Mean AUC: 0.9914 | Mean F1: 0.6936

threat and identity_hate improved by 6x over the baseline — the biggest wins came from Focal Loss forcing the model to learn from rare, hard examples.


📊 Dataset

Jigsaw Toxic Comment Classification Challenge — Kaggle


159,571 Wikipedia talk-page comments
6 binary labels per comment (multi-label — one comment can have multiple labels)
89.8% of comments are completely clean → severe class imbalance


LabelPositives% of datatoxic15,2949.6%obscene8,4495.3%insult7,8774.9%severe_toxic1,5951.0%identity_hate1,4050.9%threat4780.3%


🏗️ Model Architecture

Raw Text
   ↓
Text Cleaning (lowercase, remove URLs, IPs)
   ↓
DistilBertTokenizerFast (max_length=256)
   ↓
DistilBERT Backbone — 66M parameters (shared for all labels)
   ↓
[CLS] token → 768-dim representation
   ↓
Dropout(0.3) → Linear(768 → 6)
   ↓
Sigmoid → 6 independent probabilities
   ↓
Per-label threshold → Binary predictions

One shared DistilBERT backbone predicts all 6 labels simultaneously. Labels like obscene and insult (correlation ~0.74) benefit from shared representations — the model learns features useful for both at once.


⚙️ Key Technical Decisions

Focal Loss — standard BCE ignores rare examples. Focal Loss down-weights easy examples and forces the model to focus on hard, rare ones like threat (0.3% of data).

Per-label pos_weight — each label gets a loss weight proportional to its imbalance ratio (e.g., threat weight = 332.8, capped at 100 for stability).

Per-label threshold tuning — default 0.5 threshold is wrong for imbalanced labels. Thresholds tuned individually via grid search on the validation set.

max_length=256 — 95th percentile of comment length is 229 words. 256 covers ~93% of comments while using half the GPU memory of 512.

lr=2e-5 with warmup — large learning rates destroy pretrained DistilBERT weights (catastrophic forgetting). 10% warmup steps + linear decay keeps training stable.

📈 Training Details

ParameterValueBase modeldistilbert-base-uncasedMax sequence length256Batch size16 (train), 32 (val)OptimizerAdamW (lr=2e-5, decay=0.01)SchedulerLinear warmup (10%) + decayLossFocal Loss (α=1.0, γ=2.0) + pos_weightEpochs3 — best saved at epoch 2Dropout0.3GPUGoogle Colab T4Training time~1.5 hours

Epoch 1 → Train: 0.1146  Val: 0.0625  AUC: 0.9908
Epoch 2 → Train: 0.0608  Val: 0.0881  AUC: 0.9914  ← best model saved
Epoch 3 → Train: 0.0407  Val: 0.1209  AUC: 0.9914  ← overfitting, not saved

📦 Requirements

fastapi==0.111.0
uvicorn==0.30.0
torch==2.3.0
transformers==4.41.0
numpy==1.26.4
pydantic==2.7.0
shap==0.52.0
scikit-learn==1.5.0

👤 Author

Lakshmi Charan Yakkala
B.Tech Computer Science & Engineering — RGUKT Nuzvid

GitHub: @lakshmicharan18

you can find the trained model in my google drive because it is too much large size to push in this github account
this is google drive url : https://drive.google.com/drive/u/0/folders/1gkVsC4GbjwBTnzRtQfHmWi3T1S5cEw04
