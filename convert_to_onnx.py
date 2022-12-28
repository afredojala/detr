import torch
import torch.onnx


def main():
    onnx_model_path = "test.onnx"

    model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=False, num_classes=1)
    checkpoint = torch.load("outputs/checkpoint.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)

if __name__ == "__main__":
    main()

