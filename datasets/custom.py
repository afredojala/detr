from pathlib import Path

from .coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists()
    training_json = 'annotations.json'
    validation_json = 'annotations.json'
    PATHS = {
        "train": (root / "datasets/train", root / "datasets/Players" / training_json),
        "val": (root / "datasets/validations", root / "datasets/validations" /validation_json )
    }
    
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
