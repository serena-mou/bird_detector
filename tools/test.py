from ultralytics import YOLO

# Select pretrained model
model = YOLO('/home/matt/Birds/Training/BIRDS/MC_8L_1/weights/best.pt')

# Test model
metrics = model.val(data = "/home/matt/Birds/SmashSets/MetaSet/test.yaml")
