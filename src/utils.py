import matplotlib.pyplot as plt
import os
from .config import config


def plot_metrics(train_losses, val_losses, train_acc, val_acc):
    """
    Строит и сохраняет графики ошибок и точности обучения/валидации по эпохам.
    """
    # Номера эпох по оси X
    epochs = range(1, len(train_losses) + 1)

    # График потерь
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")  # Потери на обучении
    plt.plot(epochs, val_losses, label="Val Loss")  # Потери на валидации
    plt.title("График потерь")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config["plot_dir"], "loss.png"))  # Сохранение графика

    # График точности
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Acc")  # Точность на обучении
    plt.plot(epochs, val_acc, label="Val Acc")  # Точность на валидации
    plt.title("График точности")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config["plot_dir"], "accuracy.png"))  # Сохранение графика
