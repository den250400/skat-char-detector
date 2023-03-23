from model import Model
import torch


class Classifier:
    def __init__(self, model_path, n_classes=35):
        self.model = Model(n_classes=n_classes)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, img):
        img_tensor = torch.tensor(img).view(1, 1, img.shape[0], img.shape[1]).type(torch.float32)
        output = self.model(img_tensor)
        output = output.detach().to('cpu').numpy()
        output = output.reshape(-1).argmax()

        return output - 1

