import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from preprocess import ImageProcessor

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)  # 20 filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)  # 50 filters
         
        # Compute the flattened size automatically
        self._to_linear = None
        self._compute_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 500)  # Adjust input size
        self.fc2 = nn.Linear(500, 26)  # 26 classes

    def _compute_flattened_size(self):
        # Pass a dummy input through conv/pool layers to get the output size
        x = torch.randn(1, 1, 32, 32)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        self._to_linear = x.view(-1).size(0)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)  # Flatten for fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load and preprocess data
cmu_pie_dir = "D:/NUS/5907/CA2/PIE"
selfie_dir = "D:/NUS/5907/CA2/Raw_Selfie"
processed_selfie_dir = "D:/NUS/5907/CA2/Selfie"

subjects = range(40, 65)  # Choose 25 subjects as an example
img_processor = ImageProcessor(cmu_pie_dir, selfie_dir, processed_selfie_dir)
X_train, X_test, y_train, y_test = img_processor.get_dataset(subjects)

# X_train should be shape (num_samples, 1, 32, 32) for grayscale 32x32 images
# Reshape and normalize input data 
X_train = X_train.reshape(-1, 1, 32, 32) / 255.0
X_test = X_test.reshape(-1, 1, 32, 32) / 255.0

# Convert data to torch tensors and create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Testing and evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())  # Move to CPU before converting to numpy
        all_labels.extend(labels.cpu().numpy())  # Move to CPU before converting to numpy
test_accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {test_accuracy:.2f}')
