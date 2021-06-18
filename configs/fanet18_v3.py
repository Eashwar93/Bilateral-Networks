# fanet_custom_v3
cfg = dict(
    model_type='fanet18_v3',
    num_aux_heads=2,
    aux_output=False,
    lr_start=1e-2,
    lr_multiplier=1,
    weight_decay=5e-4,
    warmup_iters=100,
    max_iter=3200,
    im_root='./datasets/Rexroth',
    train_im_anns='./datasets/Rexroth/train.txt',
    val_im_anns='./datasets/Rexroth/eval.txt',
    scales=[0.5, 2.0],
    cropsize=[480, 640],
    ims_per_gpu=6,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
    categories=2,
    save_name='fanet18_v3.pth',
    )