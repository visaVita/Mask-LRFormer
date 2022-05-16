import imp
import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.charades_dataset import Charades
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )
    elif args.dataname == 'charades':
        dataset_dir = args.dataset_dir
        train_dataset = Charades(
            mode='train',
            data_dir=dataset_dir,
            num_ensemble_views=1,
            num_spatial_crops=1,
            data_prefix='data/Charades/Charades_rgb',
            num_frames=32,
            sample_rate=2,
            train_jitter_scales=[256, 340],
            crop_size=224,
            num_classes=157
        )
        val_dataset = Charades(
            mode='val',
            data_dir=dataset_dir,
            num_ensemble_views=1,
            num_spatial_crops=1,
            data_prefix='data/Charades/Charades_rgb',
            num_frames=32,
            sample_rate=2,
            train_jitter_scales=[256, 340],
            crop_size=256,
            num_classes=157
        )

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
