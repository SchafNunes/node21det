import numpy as np
import pytest

from node21det.evaluation import (
    Prediction,
    average_precision,
    evaluate,
    froc,
    image_scores,
    image_scores_outside_nodules,
    iou_matrix,
    match_image,
    roc_auc,
)

GT = np.array([[0, 0, 10, 10]], dtype=float)


def pred(boxes, scores):
    return Prediction(np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float))


def test_iou_identical_and_disjoint():
    ious = iou_matrix(np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), GT)
    assert ious[:, 0] == pytest.approx([1.0, 0.0])


def test_iou_exactly_0_2_is_not_a_match():
    # caixa 0..10 x 0..2 dentro de GT: interseção 20, união 100 -> IoU 0,2
    p = pred([[0, 0, 10, 2]], [0.9])
    assert iou_matrix(p.boxes, GT)[0, 0] == pytest.approx(0.2)
    assert not match_image(p, GT).any()


def test_duplicate_on_same_nodule_is_false_positive():
    p = pred([[0, 0, 10, 10], [1, 1, 10, 10]], [0.5, 0.9])
    # a de maior escore leva o nódulo, a outra vira FP
    assert match_image(p, GT).tolist() == [False, True]


def test_prediction_takes_highest_iou_unassigned_nodule():
    gts = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], float)
    p = pred([[5, 0, 15, 10], [0, 0, 10, 10]], [0.9, 0.8])
    assert match_image(p, gts).tolist() == [True, True]


def test_negative_image_all_false_positives():
    assert match_image(pred([[0, 0, 5, 5]], [0.3]), np.zeros((0, 4))).tolist() == [False]


def test_froc_counts_per_threshold():
    # imagem 1: 1 nódulo, achado com 0.9; FP com 0.4. imagem 2: negativa, FP com 0.6
    preds = [pred([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.4]), pred([[0, 0, 10, 10]], [0.6])]
    gts = [GT, np.zeros((0, 4))]
    c = froc(preds, gts)
    assert c.thresholds.tolist() == [0.9, 0.6, 0.4]
    assert c.sensitivity.tolist() == [1.0, 1.0, 1.0]
    assert c.fps_per_image.tolist() == [0.0, 0.5, 1.0]


def test_froc_tied_scores_form_one_operating_point():
    preds = [pred([[0, 0, 10, 10], [50, 50, 60, 60]], [0.5, 0.5])]
    c = froc(preds, [GT])
    assert c.thresholds.tolist() == [0.5]
    assert c.fps_per_image.tolist() == [1.0]


def test_sensitivity_interpolates_linearly_between_points():
    # 2 nódulos, 4 imagens. Pontos de operação: (0 FP/img, sens 0.5) e (0.5 FP/img, sens 1.0)
    gts = [GT, GT.copy(), np.zeros((0, 4)), np.zeros((0, 4))]
    preds = [
        pred([[0, 0, 10, 10]], [0.9]),
        pred([[0, 0, 10, 10]], [0.3]),
        pred([[0, 0, 1, 1]], [0.3]),
        pred([[0, 0, 1, 1]], [0.3]),
    ]
    c = froc(preds, gts)
    assert c.sensitivity_at(0.25) == pytest.approx(0.75)
    assert c.sensitivity_at(0.125) == pytest.approx(0.625)
    assert c.sensitivity_at(10) == pytest.approx(1.0)


def test_sensitivity_takes_best_at_same_fp_rate():
    # dois TPs seguidos sem FP entre eles: em FP/img 0 a sensibilidade é a maior
    preds = [pred([[0, 0, 10, 10]], [0.9]), pred([[0, 0, 10, 10]], [0.8])]
    c = froc(preds, [GT, GT.copy()])
    assert c.sensitivity_at(0.0) == pytest.approx(1.0)


def test_image_score_is_max_or_zero():
    assert image_scores([pred([[0, 0, 1, 1]] * 2, [0.2, 0.7]), pred([], [])]).tolist() == [0.7, 0.0]


def test_auc_with_ties():
    assert roc_auc(np.array([0.9, 0.5, 0.5, 0.1]), np.array([1, 1, 0, 0])) == pytest.approx(0.875)


def test_average_precision_perfect_and_half():
    assert average_precision([pred([[0, 0, 10, 10]], [0.9])], [GT]) == pytest.approx(1.0)
    # FP de escore maior antes do TP: precisão 0.5 no recall 1
    p = pred([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.8])
    assert average_precision([p], [GT]) == pytest.approx(0.5)


def test_rank_combines_auc_and_sensitivity():
    preds = [pred([[0, 0, 10, 10]], [0.9]), pred([], [])]
    m = evaluate(preds, [GT, np.zeros((0, 4))])
    assert m["auc"] == 1.0 and m["sens@0.25"] == 1.0
    assert m["rank"] == pytest.approx(1.0)
    assert m["mean_sens"] == pytest.approx(1.0)
    assert m["n_lesions"] == 1


def test_mean_sens_averages_the_three_fp_rates():
    # mesmo cenário da interpolação: sens 0.625, 0.75 e 1.0 em 1/8, 1/4 e 1/2
    gts = [GT, GT.copy(), np.zeros((0, 4)), np.zeros((0, 4))]
    preds = [pred([[0, 0, 10, 10]], [0.9]), pred([[0, 0, 10, 10]], [0.3]),
             pred([[0, 0, 1, 1]], [0.3]), pred([[0, 0, 1, 1]], [0.3])]
    assert evaluate(preds, gts)["mean_sens"] == pytest.approx((0.625 + 0.75 + 1.0) / 3)


def test_outside_nodules_ignores_predictions_touching_a_nodule():
    p = pred([[0, 0, 10, 10], [9, 9, 30, 30], [50, 50, 60, 60]], [0.9, 0.8, 0.3])
    # as duas primeiras tocam o nódulo (a segunda com IoU baixo, mas > 0); só a terceira conta
    assert image_scores_outside_nodules([p], [GT]).tolist() == [0.3]
    assert image_scores_outside_nodules([p], [np.zeros((0, 4))]).tolist() == [0.9]
