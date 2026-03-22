_base_ = 'mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.6
                 ),
        max_per_img=500))

train_dataloader = dict(dataset=dict(metainfo=dict(classes=('person', ))))


input_shape = 960

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_shape, input_shape), keep_ratio=True),
    dict(
        type='Pad',
        size=(input_shape, input_shape),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict( dataset=dict(pipeline=test_pipeline,metainfo=dict(classes=('person', ))))
test_dataloader = val_dataloader
