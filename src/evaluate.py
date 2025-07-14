import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import os
from .config import config


def evaluate_model(model, loader, criterion):
    """
    Оценивает модель на валидационной выборке.
    Возвращает среднюю потерю и точность.
    """
    model.eval()  # режим оценки: отключение Dropout и BatchNorm
    loss_total = 0
    correct, total = 0, 0

    with torch.no_grad():  # отключение градиентов для ускорения
        for images, labels in loader:
            images, labels = images.to(config["device"]), labels.to(config["device"])
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()  # накапливаем loss
            _, preds = torch.max(outputs, 1)  # предсказанные классы
            correct += (preds == labels).sum().item()  # количество правильных
            total += labels.size(0)  # общее количество

    # Средняя потеря и точность
    return loss_total / len(loader), correct / total


def test_model(model, test_loader, class_names):
    """
    Загружает сохранённую модель и оценивает её на тестовой выборке.
    Выводит отчёт по метрикам и сохраняет матрицу ошибок.
    """
    # Загрузка весов обученной модели
    model.load_state_dict(torch.load(config["save_path"]))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():  # отключение градиентов
        for images, labels in test_loader:
            images = images.to(config["device"])
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Печать отчёта
    print(
        "Classification Report:\n",
        classification_report(all_labels, all_preds, target_names=class_names),
    )
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(
        f"Precision: {precision_score(all_labels, all_preds, average='weighted'):.4f}"
    )
    print(f"Recall: {recall_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")

    # Построение и сохранение матрицы ошибок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(os.path.join(config["plot_dir"], "confusion_matrix.png"))
