from PIL import Image
import torch
from torchvision import transforms
from .config import config


def infer_image(model, image_path, class_names):
    """
    Выполняет инференс (предсказание) по одному изображению.
    Загружает обученную модель, обрабатывает изображение и выводит вероятности по каждому классу.
    """
    # Загружаем веса обученной модели
    model.load_state_dict(torch.load(config["save_path"]))
    model.eval()  # Переводим модель в режим инференса (без Dropout и BatchNorm)

    # Трансформация изображения — приведение к нужному размеру и преобразование в тензор
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    # Открытие изображения и приведение к RGB (на случай, если изображение не в RGB)
    image = Image.open(image_path).convert("RGB")

    # Преобразование в батч из одного изображения и отправка на нужное устройство
    input_tensor = transform(image).unsqueeze(0).to(config["device"])

    # Инференс без вычисления градиентов
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)  # Вероятности по каждому классу
        pred_class = torch.argmax(
            probs
        ).item()  # Индекс класса с наибольшей вероятностью

    # Печать предсказанного класса
    print(f"Предсказанный класс: {class_names[pred_class]}")

    # Печать вероятностей для всех классов
    for idx, prob in enumerate(probs.cpu().numpy()[0]):
        print(f"{class_names[idx]}: {prob:.4f}")
