import torch

def main():
    checkpoint = torch.load("detr-r50-e632da11.pth", map_location='cpu')
    del checkpoint['model']['class_embed.weight']
    del checkpoint['model']['class_embed.bias']
    torch.save(checkpoint, "detr-r50_no-class-head.pth")


if __name__=='__main__':
    main()
