from benchmarks import BoundingBox
from loader import Vec2


def test_iou():
    gt_box = BoundingBox(Vec2(320, 220), Vec2(680, 900))
    pred_box = BoundingBox(Vec2(500, 320), Vec2(550, 700))
    assert gt_box.intersection(pred_box) == 350000
    assert gt_box.union(pred_box) == 647000
    assert gt_box.intersection_over_union(pred_box) - 0.5409582689335394 < 0.000001
