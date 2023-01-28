import numpy as np
import base64
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = None  # this variable will contain saved model
classes = ['neutral', 'nsfw']
classes_dict = {v: k for k, v in enumerate(classes)}  # maps class label to number
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])


def get_cv2_image_from_base64_string(b64str):
    """
    Takes b64 string-encoded image, that comes from front,
    decodes it and returns image
    :param b64str: b64 encoded image
    :return: decoded image
    """
    encoded_data = b64str.split(',')[1]
    arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def prep_image(image_path, image_base64_data):
    """
    Receives either path to image or string with
    b64 representation of image and prepares to
    be sent as model input
    :param image_path: path to image
    :param image_base64_data: string containing b64 representation of image
    :return: transformed image
    """
    if image_path:
        img = np.array(Image.open(image_path).convert('RGB'))
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    img = transform(img / 255.0)
    return img


def classify_image(image_base64_data, image_path=None):
    """
    Get prediction for image
    :param image_base64_data: string containing b64 representation of image
    :param image_path: path to image
    :return: dict with predicted label, probabilities and label-number mapping
    """
    img = prep_image(image_path, image_base64_data)
    with torch.no_grad():
        out = model(torch.tensor(img.unsqueeze(0)).float().to(device))
    probabilities = torch.nn.functional.softmax(out, dim=1)
    pred_lab = np.argmax(probabilities.cpu(), axis=1).numpy()[0]

    result = {
        'class': classes[pred_lab],
        'class_probability': np.around(probabilities.tolist()[0], 4).tolist(),
        'class_dictionary': classes_dict
    }
    return result


def load_model():
    """
    Loads saved model
    """
    global model
    if model is None:
        model = torch.load('./artifacts/resnet18_nsfw.pth',
                           map_location=torch.device(device))
        model.eval()


if __name__ == "__main__":
    load_model()
    with open('b64.txt') as f:  # this file contains img30.jpg
        print(classify_image(f.read(), None))
    print(classify_image(None, './test_images/img1.jpg'))
    print(classify_image(None, './test_images/img5.jpg'))
    print(classify_image(None, './test_images/imgbad1.jpg'))
    print(classify_image(None, './test_images/imgbad15.jpg'))
