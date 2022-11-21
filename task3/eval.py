from matplotlib import pyplot as plt

from abbyy_course_cvdl_t3.coco_text import COCO_Text
from abbyy_course_cvdl_t3.utils import evaluate_ap_from_cocotext_json


def evaluate(ct: COCO_Text, pred_path: str = 'predictions.json', show_pr_curve: bool = True) -> None:
    ap, prec, rec = evaluate_ap_from_cocotext_json(
        coco_text=ct,
        path=pred_path
    )
    print(f"Итоговый скор AP на val: {ap}")

    if show_pr_curve:
        plt.figure(figsize=(12, 8))
        plt.plot(prec, rec)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.title('PR curve')
        plt.grid()
        plt.show()
