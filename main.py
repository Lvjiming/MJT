import os
import argparse
import torch
import torch.optim as optim
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from core.load_data import read_split_data, MyDataSet_1, MyDataSet_2, read_data, MyDataSet_3
from core.model_mutil import vit_gmvt_patchx_62_128 as create_model
from core.train import train_epoch, train_epoch2, train_epoch3, train_epoch4
from core.mydataset import MyDataSet, MyDataSet_3, MyDataSet_4

if __name__ == '__main__':
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    best_acc = 0
    num_classes = 9  # target of class
    epochs = 300
    batch_size = 32
    lr_up = 0.0001  # lr_up is for SGD/ADAM/cosine
    lr_down = 0.00001
    img_size = 224
    EXP_id = 4
    view_num = 3  #
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=".\\your data path")
    parser.add_argument('--Tpath', '--dataset-train-path', type=str,
                        default=".\\your data path\\EXP{}\\{}\\Train".format(EXP_id, view_num))
    parser.add_argument('--Epath', '--dataset-test-path', type=str,
                        default=".\\your data path\\EXP{}\\{}\\Test".format(EXP_id, view_num))
    parser.add_argument('--model-name', default='', help='create model name')
    tb_writer = SummaryWriter()  # used for visualisation
    parser.add_argument('--weights', type=str, default='your weight path\\premodel.pth', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)  # # Whether to freeze weights
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')  # Normal cuda: 0 is fine
    args = parser.parse_args()  # parsing parameter
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    # -----------------------------Segmentation of the data set in the same angle---------------------------------------
    # The three quantities read_split_data are the path, the proportion of the test set, and whether to draw or not #
    # The third parameter indicates whether to draw or not
    # ------------------------------------------------------------------------------------------------------------------
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path, 0.4,
    #                                                                                            False)
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------Heterogeneous Angle Segmentation of Data Sets-------------------------------------------
    train_images_path, train_images_label = read_data(args.Tpath, 'Train', False)
    val_images_path, val_images_label = read_data(args.Epath, 'Test', False)
    # ------------------------------------------------------------------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])}
    # Instantiate the training dataset, you need to use MyDataSet when you do Mstar single view,
    # and MyDataSet4 for the others
    train_dataset1 = MyDataSet_4(images_path=train_images_path,
                                 images_class=train_images_label,
                                 transform=data_transform["train"])
    val_dataset1 = MyDataSet_4(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])
    # ------------------------------------------------------------------------------------------------------------------
    # Instantiating the validation dataset
    # ------------------------------------------------------------------------------------------------------------------
    # val_dataset1 = MyDataSet(images_path=val_images_path,
    #                          images_class=val_images_label,
    #                          transform=data_transform["val"])
    # train_dataset1 = MyDataSet_3(images_path=train_images_path,
    #                              images_class=train_images_label,
    #                              transform=data_transform["train"])
    # train_dataset2 = MyDataSet_2(images_path=train_images_path,
    #                              images_class=train_images_label,
    #                              transform=data_transform["train"])
    # data_mean, data_std = std_mean(train_dataset) # 正常不用计算，因为图片多很费时
    # ------------------------------------------------------------------------------------------------------------------
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset1.collate_fn,
                                                drop_last=True)
    val_loader1 = torch.utils.data.DataLoader(val_dataset1,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=val_dataset1.collate_fn,
                                              drop_last=True)
    # ------------------------------------------------------------------------------------------------------------------
    # train_loader2 = torch.utils.data.DataLoader(train_dataset2,
    #                                             batch_size=batch_size,
    #                                             shuffle=True,
    #                                             pin_memory=True,
    #                                             num_workers=nw,
    #                                             collate_fn=train_dataset2.collate_fn)
    # val_loader2 = torch.utils.data.DataLoader(val_dataset2,
    #                                           batch_size=batch_size,
    #                                           shuffle=False,
    #                                           pin_memory=True,
    #                                           num_workers=nw,
    #                                           collate_fn=val_dataset2.collate_fn)
    # ---------------------------------Creating main Models-------------------------------------------------------------
    model = create_model(num_classes=num_classes, view_num=view_num).to(device)
    # Pre-modelling can be loaded if required
    # state_dict = torch.load('.\\your path\\vit_base_patch16_224.pth')  # Use model parameter files on local disk
    # model.load_state_dict(state_dict)  # Load the read-in model parameters into the model
    # --------------------------Creating Models ————resnet18-----------------------------------------------------------
    # model = resnet18(num_classes=num_classes, pretrained=False).to(device)
    # model = resnet18(pretrained=False).to(device)
    # images = torch.zeros(1, 3, 224, 224).to(device)  # 要求大小与输入图片的大小一致
    # tb_writer.add_graph(model, images, verbose=False)
    # ---------------------------resnet18 weight removes the full connectivity layer------------------------------------
    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #   -------------------
    #     TIPS：# Remove unwanted weights, loading weights that don't match the model will report an error,
    #     here I've removed the other two weights
    #   --------------------
    #     if model.fc:
    #         del_keys = ['fc.weight', 'fc.bias']
    #     else:
    #         del_keys = ['fc.weight', 'fc.bias']
    #
    #     for k in del_keys:
    #         del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))
    # ---------------------------vit weight removes the full connectivity layer-----------------------------------------
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        #   -------------------
        #     TIPS：# Remove unwanted weights, loading weights that don't match the model will report an error,
        #     here I've removed the other two weights
        #   --------------------
        if model.has_logits:
            del_keys = ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        else:
            del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights frozen except head, pre_logits
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    # -----------------------------optimiser----------------------------------------------------------------------------
    pg = [p for p in model.parameters() if p.requires_grad]  # pg denotes the list of loaded parameters
    # optimizer = optim.SGD(pg, lr=lr_up, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(pg, lr=lr_up, weight_decay=5E-2)
    # optimizer = optim.Adam(pg, lr=lr_up, weight_decay=0.001)
    # ----------------------------
    # cosine, which is called the annealed cosine method to reduce the learning rate, set the maximum value (1) and the
    # minimum value (lrf), and then divide it equally according to the step size, the original period of cos is 2T, now
    # in order to read the descent quickly, we set it to T
    # ----------------------------
    # lr = lambda x: ((1 + math.cos(x * math.pi / (args.epochs))) / 2) * (1 - args.lr_down) + args.lr_down
    # scheduler1 = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    # gamma = math.pow((lr_down / lr_up), 1 / epochs)
    # print("gamma:{}".format(gamma))
    # scheduler1 = lr_scheduler.StepLR(optimizer, 1, gamma=gamma, last_epoch=-1)
    # ------------------------------------------------------------------------------------------------------------------
    for epoch in range(epochs):
        print("The {} epoch的lr：{}".format(epoch, optimizer.param_groups[0]['lr']))
        # --------------------------------------------------------------------------------------------------------------
        # train
        train_loss1, train_acc1, pre_class1 = train_epoch3(model=model,
                                                           optimizer=optimizer,
                                                           data_loader=train_loader1,
                                                           device=device,
                                                           epoch=epoch, n_view=view_num)
        # test
        val_loss1, val_acc1, pre_class_val1, label_val1 = test_mutil(model=model,
                                                                     data_loader=val_loader1,
                                                                     device=device,
                                                                     epoch=epoch)
        # ------------Choosing the right type of training for different tasks-------------------------------------------
        # train_loss1, train_acc1, pre_class1 = train_epoch2(model=model,
        #                                                    optimizer=optimizer,
        #                                                    data_loader=train_loader1,
        #                                                    device=device,
        #                                                    epoch=epoch, class_num=num_classes, n_view=1)
        # val_loss1, val_acc1, pre_class_val1, label_val1 = test_4(model=model,
        #                                                              data_loader=val_loader1,
        #                                                              device=device,
        #                                                              epoch=epoch, n_view=view_num)
        # --------------------------------------------------------------------------------------------------------------
        # scheduler1.step() # Annealing cosine used

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss1, epoch)
        tb_writer.add_scalar(tags[1], train_acc1, epoch)
        tb_writer.add_scalar(tags[2], val_loss1, epoch)
        tb_writer.add_scalar(tags[3], val_acc1, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(model.state_dict(), "./weights/EXP{}_{}_best_acc.pth".format(EXP_id, view_num))
        print("The Best Acc = : {:.4f}".format(best_acc))
