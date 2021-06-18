import torch


if __name__ =="__main__":
    ignore = 255
    hist = torch.zeros(3, 3, dtype=int).cuda().detach()
    print("hist :",hist)
    label = torch.ones(2,2, dtype=int).cuda().detach()
    label[[0], [0]] = 0
    label[[0], [1]] = 2
    keep = label != ignore
    print("keep", keep)
    print("label :", label)
    preds = torch.ones(2,2,dtype=int).cuda().detach()
    preds[[0], [0]] = 0
    preds[[0], [1]] = 2

    print("label[keep]: ", label[keep])
    print("preds[keep]: ", preds[keep])

    hist += torch.bincount(label[keep] * 3 + preds[keep], minlength=3**2).view(3,3)

    print("hist", hist)

    print("hist.diag", hist.diag())
    print("hist.sum.dim0", hist.sum(dim=0))
    print("hist.sum.dim1", hist.sum(dim=1))

    input = label[keep] * 3 + preds[keep]
    print(torch.bincount(input, minlength=9))



