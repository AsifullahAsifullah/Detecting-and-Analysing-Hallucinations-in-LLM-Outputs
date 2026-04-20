# tests/quick_test.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from utils import normalize_text, format_input, set_seed
from config import SETTINGS
from build_dataset import balance_dataset

_results = {"passed": 0, "failed": 0}
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"

def check(label, condition):
    if condition:
        print(f"  [{PASS}] {label}")
        _results["passed"] += 1
    else:
        print(f"  [{FAIL}] {label}")
        _results["failed"] += 1

def info(label):
    print(f"  [{INFO}] {label}")


def test_utils():
    print("\n[1] Utils")
    check("normalize_text strips whitespace",  normalize_text("  hello   world  ") == "hello world")
    check("normalize_text handles empty",      normalize_text("") == "")
    check("format_input with all fields",      "[SEP]" in format_input("answer", "question", "knowledge"))
    check("format_input answer only",          "Answer: hello" in format_input("hello"))
    check("format_input includes knowledge",   "Knowledge:" in format_input("a", knowledge="k"))
    check("format_input includes question",    "Question:" in format_input("a", question="q"))
    set_seed(42)
    check("set_seed runs without error", True)


def test_config():
    print("\n[2] Config")
    check("BERT model name defined",    "bert"    in SETTINGS.model.architectures)
    check("RoBERTa model name defined", "roberta" in SETTINGS.model.architectures)
    check("DeBERTa model name defined", "deberta" in SETTINGS.model.architectures)
    check("max_seq_length > 0",         SETTINGS.model.max_seq_length > 0)
    check("num_labels == 2",            SETTINGS.model.num_labels == 2)
    check("splits sum to 1.0",
          abs(SETTINGS.training.train_size +
              SETTINGS.training.val_size   +
              SETTINGS.training.test_size  - 1.0) < 1e-6)
    check("learning_rate > 0", SETTINGS.training.learning_rate > 0)
    check("num_epochs > 0",    SETTINGS.training.num_epochs > 0)


def test_dataset_utils():
    print("\n[3] Dataset Utilities")

    rows     = [{"label": 1}] * 8 + [{"label": 0}] * 4
    balanced = balance_dataset(rows)
    labels   = [r["label"] for r in balanced]
    check("balance_dataset equalises classes", labels.count(0) == labels.count(1))
    check("balance_dataset returns list",       isinstance(balanced, list))

    long_text = "word " * 300
    formatted = format_input(long_text, "question?", "knowledge context " * 50)
    check("format_input handles long text", len(formatted) > 100)

    from build_dataset import load_truthfulqa_csv
    import tempfile, csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Best Answer"])
        for i in range(10):
            writer.writerow([f"Q{i}?", f"Answer {i}"])
        tmp_path = f.name
    rows_tqa = load_truthfulqa_csv(tmp_path)
    os.unlink(tmp_path)
    check("load_truthfulqa_csv returns list",  isinstance(rows_tqa, list))
    check("load_truthfulqa_csv has 10 rows",   len(rows_tqa) == 10)
    check("load_truthfulqa_csv labels are 0",  all(r["label"] == 0 for r in rows_tqa))
    check("load_truthfulqa_csv has question",  all("question" in r for r in rows_tqa))

    # split_dataset — only test if torch is available
    try:
        import torch
        from dataset import split_dataset
        dummy = [{"label": i % 2, "text": f"sample {i}"} for i in range(100)]
        train, val, test = split_dataset(dummy, random_seed=42)
        check("split_dataset train ~60%", 55 <= len(train) <= 65)
        check("split_dataset val  ~20%",  15 <= len(val)   <= 25)
        check("split_dataset test ~20%",  15 <= len(test)  <= 25)
        check("split_dataset no overlap",
              len(set(i["text"] for i in train) &
                  set(i["text"] for i in test)) == 0)
    except ImportError:
        info("split_dataset test skipped (torch not installed)")


def test_imports():
    print("\n[4] Library Checks (informational)")
    try:
        import torch
        check("torch available", True)
        from transformers import AutoTokenizer, AutoModel
        check("transformers available", True)
        from pipeline import DeepLearningHallucinationPipeline
        check("DeepLearningHallucinationPipeline importable", True)
        check("has predict method",     hasattr(DeepLearningHallucinationPipeline, "predict"))
        check("has predict_row method", hasattr(DeepLearningHallucinationPipeline, "predict_row"))
        check("has load method",        hasattr(DeepLearningHallucinationPipeline, "load"))
        from model import TransformerClassifier
        check("TransformerClassifier importable", True)
    except ImportError as e:
        info(f"torch/transformers not installed: {e}")
        info("Run: pip install -r requirements.txt")
        info("Core tests (sections 1-3) pass independently of torch.")


if __name__ == "__main__":
    print("=" * 55)
    print("  Deep Learning Detector: Quick Tests")
    print("=" * 55)

    test_utils()
    test_config()
    test_dataset_utils()
    test_imports()

    total = _results["passed"] + _results["failed"]
    print(f"\n{'='*55}")
    print(f"  Results: {_results['passed']}/{total} passed")
    print("  All tests passed!" if _results["failed"] == 0 else f"  {_results['failed']} test(s) failed.")
    print("=" * 55)
    sys.exit(0 if _results["failed"] == 0 else 1)
