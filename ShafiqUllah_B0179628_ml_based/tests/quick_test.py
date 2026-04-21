# tests/quick_test.py


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import traceback

PASS = "✓"
FAIL = "✗"
results = []


def test(name: str, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))


def test_lexical_basic():
    from modules.lexical_features import extract_lexical_features
    feats = extract_lexical_features("The capital of Germany is Madrid.")
    assert isinstance(feats, dict)
    assert "lex_word_count" in feats
    assert "lex_hedge_ratio" in feats
    assert all(0.0 <= v <= 1.0 for v in feats.values()), f"Out of range: {feats}"


def test_lexical_empty():
    from modules.lexical_features import extract_lexical_features
    feats = extract_lexical_features("")
    assert all(v == 0.0 for v in feats.values())


def test_lexical_hedge():
    from modules.lexical_features import extract_lexical_features
    feats_hedge = extract_lexical_features("Maybe this is perhaps possibly true.")
    feats_plain = extract_lexical_features("This is definitely true.")
    assert feats_hedge["lex_hedge_ratio"] > feats_plain["lex_hedge_ratio"]


def test_semantic_basic():
    from modules.semantic_features import extract_semantic_features
    feats = extract_semantic_features(
        answer="Berlin is the capital of Germany.",
        question="What is the capital of Germany?",
        knowledge="Germany's capital is Berlin.",
    )
    assert isinstance(feats, dict)
    assert "sem_ans_knowledge_sim" in feats
    assert feats["sem_ans_knowledge_sim"] > 0.0


def test_semantic_dissimilarity():
    from modules.semantic_features import extract_semantic_features
    feats_wrong = extract_semantic_features(
        answer="The capital is Madrid.",
        knowledge="Germany's capital is Berlin.",
    )
    feats_right = extract_semantic_features(
        answer="The capital is Berlin.",
        knowledge="Germany's capital is Berlin.",
    )
    assert feats_wrong["sem_knowledge_dissimilarity"] >= feats_right["sem_knowledge_dissimilarity"]


def test_semantic_empty():
    from modules.semantic_features import extract_semantic_features
    feats = extract_semantic_features("")
    assert all(v == 0.0 for v in feats.values())


def test_linguistic_basic():
    from modules.linguistic_features import extract_linguistic_features
    feats = extract_linguistic_features(
        "The quick brown fox jumps over the lazy dog. "
        "This sentence was written passively. "
        "It contains multiple clauses, which adds complexity."
    )
    assert isinstance(feats, dict)
    assert "ling_fk_grade" in feats
    assert "ling_passive_ratio" in feats
    assert all(0.0 <= v <= 1.0 for v in feats.values())


def test_linguistic_passive():
    from modules.linguistic_features import extract_linguistic_features
    feats_passive = extract_linguistic_features(
        "The book was written by the author. The report was prepared by the team."
    )
    feats_active = extract_linguistic_features(
        "The author wrote the book. The team prepared the report."
    )
    assert feats_passive["ling_passive_ratio"] >= feats_active["ling_passive_ratio"]


def test_linguistic_empty():
    from modules.linguistic_features import extract_linguistic_features
    feats = extract_linguistic_features("")
    assert all(v == 0.0 for v in feats.values())


def test_confidence_basic():
    from modules.confidence_features import extract_confidence_features
    feats = extract_confidence_features(
        "This is definitely absolutely certainly the correct answer."
    )
    assert isinstance(feats, dict)
    assert "conf_certainty_ratio" in feats
    assert feats["conf_certainty_ratio"] > 0.0


def test_confidence_hedge():
    from modules.confidence_features import extract_confidence_features
    feats_hedge = extract_confidence_features(
        "Maybe perhaps this could possibly be true, it seems."
    )
    feats_plain = extract_confidence_features(
        "This is correct."
    )
    assert feats_hedge["conf_hedge_ratio"] > feats_plain["conf_hedge_ratio"]


def test_confidence_empty():
    from modules.confidence_features import extract_confidence_features
    feats = extract_confidence_features("")
    assert all(v == 0.0 for v in feats.values())


def test_feature_pipeline_fit_transform():
    from feature_pipeline import FeaturePipeline

    sample_rows = [
        {"text": "The capital of Germany is Berlin.", "question": "Capital of Germany?", "knowledge": "Berlin is capital.", "label": 0},
        {"text": "The capital of Germany is Madrid.", "question": "Capital of Germany?", "knowledge": "Berlin is capital.", "label": 1},
        {"text": "Water boils at 100 degrees.", "question": "Boiling point?", "knowledge": "100C", "label": 0},
        {"text": "Water boils at 500 degrees.", "question": "Boiling point?", "knowledge": "100C", "label": 1},
    ]

    fp = FeaturePipeline(tfidf_max_features=100, tfidf_ngram_range=(1, 1))
    X = fp.fit_transform(sample_rows)
    assert X.shape[0] == 4
    assert X.shape[1] > 10  # should have many features


def test_feature_pipeline_transform():
    from feature_pipeline import FeaturePipeline
    import numpy as np

    train_rows = [
        {"text": "Berlin is the capital.", "question": "", "knowledge": "", "label": 0},
        {"text": "Madrid is the capital.", "question": "", "knowledge": "", "label": 1},
    ]
    test_rows = [
        {"text": "Paris is the capital.", "question": "", "knowledge": "", "label": 0},
    ]

    fp = FeaturePipeline(tfidf_max_features=50, tfidf_ngram_range=(1, 1))
    X_train = fp.fit_transform(train_rows)
    X_test = fp.transform(test_rows)

    assert X_test.shape[1] == X_train.shape[1]


def test_combined_features():
    from feature_pipeline import extract_handcrafted_features
    feats = extract_handcrafted_features(
        answer="Napoleon used the internet in 1804.",
        question="How did Napoleon communicate?",
        knowledge="Napoleon used messengers and written orders.",
    )
    assert len(feats) >= 40  # at least 40 handcrafted features
    assert all(isinstance(v, float) for v in feats.values())


if __name__ == "__main__":
    print("=" * 60)
    print("  QUICK TEST SUITE — ML Pipeline")
    print("=" * 60)

    print("\n  Lexical Features:")
    test("basic extraction", test_lexical_basic)
    test("empty input", test_lexical_empty)
    test("hedge words detected", test_lexical_hedge)

    print("\n  Semantic Features:")
    test("basic extraction", test_semantic_basic)
    test("dissimilarity higher for wrong answer", test_semantic_dissimilarity)
    test("empty input", test_semantic_empty)

    print("\n  Linguistic Features:")
    test("basic extraction", test_linguistic_basic)
    test("passive voice detected", test_linguistic_passive)
    test("empty input", test_linguistic_empty)

    print("\n  Confidence Features:")
    test("certainty words detected", test_confidence_basic)
    test("hedge words detected", test_confidence_hedge)
    test("empty input", test_confidence_empty)

    print("\n  Feature Pipeline:")
    test("fit_transform", test_feature_pipeline_fit_transform)
    test("transform consistency", test_feature_pipeline_transform)
    test("combined features count", test_combined_features)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} tests passed")
    if passed == total:
        print("  All tests passed! ✓")
    else:
        print("  Failed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"    - {name}: {err}")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)
