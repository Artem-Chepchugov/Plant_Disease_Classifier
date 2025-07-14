import torch.nn as nn
from .config import config


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Последовательная сверточная нейросеть (CNN), состоящая из 3 блоков:
        self.net = nn.Sequential(
            # Блок 1: Свертка → ReLU → MaxPooling → Dropout
            nn.Conv2d(
                3, 32, 3, padding=1
            ),  # Вход: 3 канала (RGB), выход: 32 канала, ядро 3x3
            nn.ReLU(),  # Нелинейность
            nn.MaxPool2d(2),  # Уменьшение размерности в 2 раза
            nn.Dropout(0.25),  # Отсев 25% нейронов
            # Блок 2: Свертка → ReLU → MaxPooling → Dropout
            nn.Conv2d(32, 64, 3, padding=1),  # 64 канала на выходе
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Блок 3: Свертка → ReLU → MaxPooling → Dropout
            nn.Conv2d(64, 128, 3, padding=1),  # 128 каналов на выходе
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Переход к полносвязным слоям:
            nn.Flatten(),  # Преобразуем 3D-выход в вектор
            nn.Linear((config["image_size"] // 8) ** 2 * 128, 256),  # Первый FC-слой
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 50% перед классификацией
            nn.Linear(
                256, num_classes
            ),  # Финальный FC-слой: выход — количество классов
        )

    def forward(self, x):
        # Прямой проход входного изображения через сеть
        return self.net(x)
