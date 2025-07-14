# Импорт необходимых библиотек и функций
import argparse  # для парсинга аргументов командной строки
from src.config import config  # глобальная конфигурация проекта
from src.dataloader import prepare_dataloaders  # функция подготовки DataLoader'ов
from src.model import SimpleCNN  # архитектура модели
from src.train import train_model  # функция обучения
from src.evaluate import test_model  # функция оценки на тестовой выборке
from src.infer import infer_image  # функция инференса по одному изображению
import torch
import torch.nn as nn
import torch.optim as optim

# Точка входа в программу
if __name__ == "__main__":
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "infer"],
        required=True,
        help="Режим работы: train — обучение, eval — тестирование, infer — инференс одного изображения",
    )
    parser.add_argument(
        "--image_path", type=str, help="Путь к изображению (только для режима infer)"
    )
    args = parser.parse_args()

    # Загрузка данных и получение DataLoader'ов + названия классов + веса классов
    train_loader, val_loader, test_loader, class_names, class_weights = (
        prepare_dataloaders()
    )

    # Инициализация модели и перемещение её на нужное устройство (CPU или GPU)
    model = SimpleCNN(num_classes=len(class_names)).to(config["device"])

    # Создание функции потерь с учётом весов классов (для компенсации дисбаланса классов)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Создание оптимизатора (Adam)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Обработка выбранного пользователем режима
    if args.mode == "train":
        # Режим обучения модели
        train_model(
            model, train_loader, val_loader, criterion, optimizer, config["num_epochs"]
        )

    elif args.mode == "eval":
        # Режим оценки на тестовой выборке
        test_model(model, test_loader, class_names)

    elif args.mode == "infer":
        # Режим инференса одного изображения (обязательно указать путь к изображению)
        if not args.image_path:
            raise ValueError(
                "Для режима infer необходимо указать путь к изображению (--image_path)"
            )
        infer_image(model, args.image_path, class_names)
