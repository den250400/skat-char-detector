from model import Model
import torch


class Classifier:
    def __init__(self, model_path, n_classes=35):
        self.model = Model(n_classes=n_classes)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, img):
        with torch.no_grad():
            img_tensor = torch.tensor(img).view(1, 1, img.shape[0], img.shape[1]).type(torch.float32)
            output = self.model(img_tensor)
            confidence = torch.nn.Sigmoid()(output[0, 0]).to('cpu').item()
            class_scores = torch.nn.Softmax()(output[0, 1:]).to('cpu').numpy()

        class_prediction = class_scores.argmax()
        print(class_scores.max())

        return confidence, class_prediction

