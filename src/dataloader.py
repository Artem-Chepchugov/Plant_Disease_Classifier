from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from collections import Counter
from .config import config


def prepare_dataloaders():
    # Аугментации и трансформации для обучающей выборки
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_size"], config["image_size"])
            ),  # изменение размера
            transforms.RandomHorizontalFlip(),  # случайное горизонтальное отражение
            transforms.RandomRotation(10),  # случайный поворот на 10 градусов
            transforms.ToTensor(),  # преобразование в тензор
        ]
    )

    # Трансформации для валидации и теста (без аугментаций)
    test_transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    # Загрузка всего датасета (в формате ImageFolder)
    full_dataset = datasets.ImageFolder(
        root=config["dataset_path"], transform=train_transform
    )

    # Подсчёт размеров выборок
    total_size = len(full_dataset)
    val_size = int(config["val_split"] * total_size)
    test_size = int(config["test_split"] * total_size)
    train_size = total_size - val_size - test_size

    # Разделение полного датасета на обучающий, валидационный и тестовый
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(
            42
        ),  # фиксированный seed для воспроизводимости
    )

    # Установка других трансформаций для валидации и теста (без аугментаций)
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # Создание DataLoader-ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # перемешивание
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # Подсчёт количества примеров каждого класса в обучающей выборке
    targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(targets)
    total_samples = sum(class_counts.values())

    # Расчёт весов классов: чем реже класс, тем больше его вес
    class_weights = [0.0] * len(full_dataset.classes)
    for class_idx in class_counts:
        class_weights[class_idx] = total_samples / class_counts[class_idx]

    # Перевод весов в тензор и перенос на нужное устройство (CPU/GPU)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(config["device"])

    # Возвращаем готовые загрузчики данных и веса классов
    return train_loader, val_loader, test_loader, full_dataset.classes, weight_tensor
