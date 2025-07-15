import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Configuration & Paths
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "chest_xray" 

train_path = os.path.join(data_dir, "train")
val_path = os.path.join(data_dir, "val")
test_path = os.path.join(data_dir, "test")

# -------------------------------
# 2. Data Transformations
# -------------------------------
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
}

# -------------------------------
# 3. Load Datasets
# -------------------------------
train_data = datasets.ImageFolder(train_path, transform=transform['train'])
val_data = datasets.ImageFolder(val_path, transform=transform['val'])
test_data = datasets.ImageFolder(test_path, transform=transform['test'])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

class_names = train_data.classes

# -------------------------------
# 4. CNN Model Definition
# -------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, x):
        return self.net(x)

model = CNNModel().to(device)

# -------------------------------
# 5. Loss, Optimizer, Scheduler
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 6. Training Loop
# -------------------------------
def train(model, loader):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# -------------------------------
# 7. Evaluation Function
# -------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = correct / total
    return acc, y_true, y_pred

# -------------------------------
# 8. Run Training
# -------------------------------
epochs = 5
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_acc, _, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

# -------------------------------
# 9. Evaluate on Test Set
# -------------------------------
test_acc, y_true, y_pred = evaluate(model, test_loader)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------------------------------
# 10. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
