from ultralytics import YOLO

# Select pretrained model
model = YOLO('yolov8l.pt')

# Start training
metrics = model.train(data = "/home/matt/Birds/SmashSets/MetaSet/data.yaml",
        epochs = 500,
        patience = 30,
        batch = -1,
        imgsz = 512,
        save = True,
        device = 0,
        workers = 12,
        project = 'BIRDS',
        name = "MC_8L_1",
        val = True,
        lr0 = 0.01,				# (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        lrf = 0.01,				# (float) final learning rate (lr0 * lrf)
        momentum = 0.937,			# (float) SGD momentum/Adam beta1
        weight_decay = 0.0005,			# (float) optimizer weight decay 5e-4
        warmup_epochs = 3.0,			# (float) warmup epochs (fractions ok)
        warmup_momentum = 0.8,			# (float) warmup initial momentum
        warmup_bias_lr = 0.1,			# (float) warmup initial bias lr
        box = 7.5,				# (float) box loss gain
        cls = 0.5,				# (float) cls loss gain (scale with pixels)
        dfl = 1.5,				# (float) dfl loss gain
        pose = 12.0,				# (float) pose loss gain
        kobj = 1.0,				# (float) keypoint obj loss gain
        label_smoothing = 0.0,			# (float) label smoothing (fraction)
        nbs = 64,				# (int) nominal batch size
        hsv_h = 0.015,				# (float) image HSV-Hue augmentation (fraction)
        hsv_s = 0.7,				# (float) image HSV-Saturation augmentation (fraction)
        hsv_v = 0.4,				# (float) image HSV-Value augmentation (fraction)
        degrees = 0.0,				# (float) image rotation (+/- deg)
        translate = 0.0,			# (float) image translation (+/- fraction)
        scale = 0.2,				# (float) image scale (+/- gain)
        shear = 0.0,				# (float) image shear (+/- deg)
        perspective = 0.0,			# (float) image perspective (+/- fraction), range 0-0.001
        flipud = 0.5,				# (float) image flip up-down (probability)
        fliplr = 0.5,				# (float) image flip left-right (probability)
        mosaic = 0.0,				# (float) image mosaic (probability)
        mixup = 0.0,				# (float) image mixup (probability)
        copy_paste = 0.0,			# (float) segment copy-paste (probability)
        auto_augment = "randaugment",		# (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
        erasing = 0.0,				# (float) probability of random erasing during classification training (0-1)
        crop_fraction = 1.0			# (float) image crop fraction for classification evaluation/inference (0-1)
)
