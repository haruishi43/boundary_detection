#!/usr/bin/env python3

import os
from tqdm import tqdm

import torch
from torch.utils import data
from torch.nn.parallel import DataParallel
from torchvision import transforms

from sbdet.datasets import get_edge_dataset, test_batchify_fn
from sbdet.models import get_edge_model
from sbdet.visualize import visualize_prediction

from scripts.option import Options


def test(args):
    # data transforms
    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    output_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
        ]
    )

    # dataset
    if args.eval:  # set split='val' for validation set testing
        testset = get_edge_dataset(
            args.dataset,
            split="val",
            mode="testval",
            transform=input_transform,
            crop_size=args.crop_size,
        )
    else:  # set split='vis' for visulization
        testset = get_edge_dataset(
            args.dataset,
            split="vis",
            mode="vis",
            transform=input_transform,
            crop_size=args.crop_size,
        )

    # output folder
    if args.eval:
        outdir_list_side5 = []
        outdir_list_fuse = []
        for i in range(testset.num_class):
            outdir_side5 = "results/boundary/%s/%s/%s_val/side5/class_%03d" % (
                args.dataset,
                args.model,
                args.checkname,
                i + 1,
            )
            if not os.path.exists(outdir_side5):
                os.makedirs(outdir_side5)
            outdir_list_side5.append(outdir_side5)

            outdir_fuse = "results/boundary/%s/%s/%s_val/fuse/class_%03d" % (
                args.dataset,
                args.model,
                args.checkname,
                i + 1,
            )
            if not os.path.exists(outdir_fuse):
                os.makedirs(outdir_fuse)
            outdir_list_fuse.append(outdir_fuse)

    else:
        outdir = "results/boundary/%s/%s/%s_vis" % (
            args.dataset,
            args.model,
            args.checkname,
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
    )
    test_data = data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=test_batchify_fn,
        **loader_kwargs
    )

    model = get_edge_model(
        args.model,
        dataset=args.dataset,
        backbone=args.backbone,
        norm_layer=torch.nn.BatchNorm2d,
        crop_size=args.crop_size,
    )

    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    if args.cuda:
        model = DataParallel(model.cuda())
    # print(model)

    model.eval()
    tbar = tqdm(test_data)

    if args.eval:
        for i, (images, im_paths, im_sizes) in enumerate(tbar):
            with torch.no_grad():
                images = [image.unsqueeze(0) for image in images]
                images = torch.cat(images, 0)
                outputs = model(images.float())
                b_side5 = outputs[0]
                b_fuse = outputs[1]

                # num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
                # if num_gpus == 1:
                #     outputs = [outputs]

                # extract the side5 output and fuse output from outputs
                side5_list = []
                fuse_list = []
                # for i in range(len(outputs)):  # iterate for n (gpu counts)
                for i, (fuse, side5) in enumerate(zip(b_fuse, b_side5)):
                    im_size = tuple(im_sizes[i].numpy())
                    # output = outputs[i]

                    side5 = side5.squeeze_()
                    # side5 = side5.sigmoid_().cpu().numpy()
                    # side5 = side5[:, 0 : im_size[1], 0 : im_size[0]]
                    side5 = side5.sigmoid_().cpu()
                    side5 = side5[:, 0 : im_size[1], 0 : im_size[0]]

                    fuse = fuse.squeeze_()
                    # fuse = fuse.sigmoid_().cpu().numpy()
                    # fuse = fuse[:, 0 : im_size[1], 0 : im_size[0]]
                    fuse = fuse.sigmoid_().cpu()
                    fuse = fuse[:, 0 : im_size[1], 0 : im_size[0]]

                    side5_list.append(side5)
                    fuse_list.append(fuse)

                for predict, impath in zip(side5_list, im_paths):
                    for i in range(predict.shape[0]):
                        predict_c = predict[i]
                        path = os.path.join(outdir_list_side5[i], impath)
                        # io.imsave(path, predict_c)
                        out = output_transform(predict_c)
                        out.save(path)

                for predict, impath in zip(fuse_list, im_paths):
                    for i in range(predict.shape[0]):
                        predict_c = predict[i]
                        path = os.path.join(outdir_list_fuse[i], impath)
                        # io.imsave(path, predict_c)
                        out = output_transform(predict_c)
                        out.save(path)

    else:
        for i, (images, masks, im_paths, im_sizes) in enumerate(tbar):
            with torch.no_grad():
                images = [image.unsqueeze(0) for image in images]
                images = torch.cat(images, 0)
                outputs = model(images.float())  # (tensor(b, c, h, w), tensor(b, c, h, w))
                b_side5 = outputs[0]
                b_fuse = outputs[1]

                # num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
                # if num_gpus == 1:
                #     outputs = [outputs]

                # extract the side5 output and fuse output from outputs
                side5_list = []
                fuse_list = []
                for i, (fuse, side5) in enumerate(zip(b_fuse, b_side5)):  # iterate for n (gpu counts)
                    im_size = tuple(im_sizes[i].numpy())

                    side5 = side5.squeeze_()
                    side5 = side5.sigmoid_().cpu().numpy()
                    side5 = side5[:, 0 : im_size[1], 0 : im_size[0]]

                    fuse = fuse.squeeze_()
                    fuse = fuse.sigmoid_().cpu().numpy()
                    fuse = fuse[:, 0 : im_size[1], 0 : im_size[0]]

                    side5_list.append(side5)
                    fuse_list.append(fuse)

                # visualize ground truth
                for gt, impath in zip(masks, im_paths):
                    outname = os.path.splitext(impath)[0] + "_gt.png"
                    path = os.path.join(outdir, outname)
                    visualize_prediction(args.dataset, path, gt)

                # visualize side5 output
                for predict, impath in zip(side5_list, im_paths):
                    outname = os.path.splitext(impath)[0] + "_side5.png"
                    path = os.path.join(outdir, outname)
                    visualize_prediction(args.dataset, path, predict)

                # visualize fuse output
                for predict, impath in zip(fuse_list, im_paths):
                    outname = os.path.splitext(impath)[0] + "_fuse.png"
                    path = os.path.join(outdir, outname)
                    visualize_prediction(args.dataset, path, predict)


def eval_model(args):
    if args.resume_dir is None:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))

    if os.path.splitext(args.resume_dir)[1] == ".tar":
        args.resume = args.resume_dir
        assert os.path.exists(args.resume_dir)
        test(args)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    eval_model(args)
