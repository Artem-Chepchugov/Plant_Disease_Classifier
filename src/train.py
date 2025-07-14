import torch
from .evaluate import (
    evaluate_model,
)
from .utils import plot_metrics
from .config import config


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0  # Лучшая достигнутая точность на валидации
    train_losses, val_losses = [], []  # Для логирования потерь по эпохам
    train_accuracies, val_accuracies = [], []  # Для логирования точности по эпохам

    for epoch in range(num_epochs):
        model.train()  # Перевод модели в режим обучения
        running_loss = 0  # Суммарная ошибка на текущей эпохе
        correct, total = 0, 0  # Подсчёт точности

        for images, labels in train_loader:
            # Перенос данных на GPU или CPU
            images, labels = images.to(config["device"]), labels.to(config["device"])

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)

            # Вычисление ошибки и обратное распространение
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Подсчёт предсказаний
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Метрики для обучения
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Метрики для валидации
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Вывод текущей эпохи
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Сохраняем модель, если валидационная точность улучшилась
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), config["save_path"])

    # Построение и сохранение графиков
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
