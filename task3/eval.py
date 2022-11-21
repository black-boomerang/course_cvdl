import os

from matplotlib import pyplot as plt

from abbyy_course_cvdl_t3 import coco_evaluation
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


def show_errors(ct: COCO_Text, imgs_dir: str, pred_path: str = 'predictions.json') -> None:
    results = ct.loadRes(str(pred_path))

    matches = coco_evaluation.getDetections(
        ct,
        results,
        imgIds=ct.val,
        score_threshold=0.5,
        area_fraction_threshold=1. / 32 / 32
    )

    fp_examples = matches['false_positives'][0:10]
    fn_examples = matches['false_negatives'][0:10]

    plt.figure(figsize=(16, 60))
    for i, (fp_example, fn_example) in enumerate(zip(fp_examples, fn_examples)):
        plt.subplot(10, 2, i * 2 + 1)
        ann = results.loadAnns(fp_example['eval_id'])[0]
        img = results.loadImgs(ann['image_id'])[0]
        plt.imshow(plt.imread(os.path.join(imgs_dir, img['file_name'])))
        results.showAnns([ann])
        plt.axis('off')
        if i == 0:
            plt.title('False Positives')

        plt.subplot(10, 2, i * 2 + 2)
        ann = ct.loadAnns(fn_example['gt_id'])[0]
        img = ct.loadImgs(ann['image_id'])[0]
        plt.imshow(plt.imread(os.path.join(imgs_dir, img['file_name'])))
        ct.showAnns([ann])
        plt.axis('off')
        if i == 0:
            plt.title('False Negatives')

    plt.tight_layout()
    plt.show()
