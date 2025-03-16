from transformers import EvalPrediction
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    y_true = eval_preds.label_ids
    y_pred = eval_preds.predictions.argmax(-1)

    return {
        'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
        'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
        #'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
    }