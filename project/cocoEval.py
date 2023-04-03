from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ground_truth_path = "/home/muradek/Deep_project/project/annotations_test.json"
output_path = "/home/muradek/Deep_project/runs/detect/val20/predictions.json"

#load annotation files
cocoGt=COCO(ground_truth_path)
cocoDt=cocoGt.loadRes(output_path)

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()