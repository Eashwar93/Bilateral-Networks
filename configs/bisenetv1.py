# biisenet v1
cfg = dict(
    model_type='bisenetv1',
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,
    im_root='./datasets/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    scales=[0.75,1. ],
    cropsize=[640, 480],
    ims_per_gpu=3,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
    )