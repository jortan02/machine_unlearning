import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import tqdm


class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.ensemble_size = len(models)
        self.ensemble = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack(tuple(model(x) for model in self.ensemble)).mean(0)


def main():
    with open("ensemble_training_results.csv", "w") as file:
        file.write("model,chunk,valid,epoch,loss,accuracy\n")

    with open("ensemble_evaluation_results.csv", "w") as file:
        file.write("chunk,n_retrained,loss,accuracy\n")

    start = time.time()
    learning_rate = 0.001
    batch_size = 512
    chunks = 5
    subset_ratio = 0.2
    num_models = 5
    deletion_ratio = 0.0001
    epochs = 50

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """'Get data"""
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    train_set_chunks_data = (
        (
            torch.tensor(train_set.data).permute(0, 3, 1, 2) / 255
            - torch.tensor([0.4914, 0.4821, 0.4465])[None, :, None, None]
        )
        / torch.tensor([0.2470, 0.2435, 0.2616])[None, :, None, None]
    ).chunk(chunks)
    train_set_chunks_targets = torch.tensor(train_set.targets).chunk(chunks)

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6)

    models = [None] * num_models
    valid = [False] * num_models
    data_usages = [None] * num_models
    all_deleted_indices = set()
    working_train_set_data = train_set_chunks_data[0]
    working_train_set_targets = train_set_chunks_targets[0]
    print("Chunk:", 1)

    for chunk_number in range(chunks):
        training_size = len(working_train_set_data)
        sample_weights = torch.ones(training_size) / (training_size - len(all_deleted_indices))
        for deleted_index in all_deleted_indices:
            sample_weights[deleted_index] = 0
        sample_size = int(subset_ratio * training_size)
        deletion_size = int(deletion_ratio * training_size)
        """Model training phase"""
        for model_index in range(num_models):
            print(f"Model {model_index + 1}, Valid:", valid[model_index])
            """Bagging training samples"""
            train_sample_indices = sample_weights.multinomial(num_samples=sample_size, replacement=True)
            train_set_samples = data.TensorDataset(
                working_train_set_data[train_sample_indices], working_train_set_targets[train_sample_indices]
            )
            """Creating the model"""
            if not valid[model_index]:
                model = torchvision.models.resnet18(num_classes=10).to(device)
                models[model_index] = model
                valid[model_index] = True
                retrain = True
                data_usages[model_index] = set(np.unique(train_sample_indices.numpy()))
            else:
                model = models[model_index]
                data_usages[model_index].update(np.unique(train_sample_indices.numpy()).tolist())
                retrain = False
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            train_loader = torch.utils.data.DataLoader(
                train_set_samples, batch_size=batch_size, shuffle=True, num_workers=6
            )
            train_individual(
                model,
                optimizer,
                criterion,
                train_loader,
                test_loader,
                model_index,
                chunk_number,
                not retrain,
                epochs,
                device,
            )
            """Setting model to ensemble"""
            data_usages[model_index] = set(np.unique(train_sample_indices.numpy()))

        evaluate_ensemble(device, test_loader, models, valid, chunk_number, criterion)

        """Deletion request phase: completed in the next loop"""
        if chunk_number == chunks - 1:
            break
        print("Data Deletions:", deletion_size)
        deletion_indices = sample_weights.multinomial(num_samples=deletion_size, replacement=False)
        for deletion_index in deletion_indices:
            all_deleted_indices.add(int(deletion_index))

            for model_index in range(num_models):
                if int(deletion_index) in data_usages[model_index]:
                    valid[model_index] = False

        working_train_set_data = torch.cat((working_train_set_data, train_set_chunks_data[chunk_number + 1]))
        working_train_set_targets = torch.cat((working_train_set_targets, train_set_chunks_targets[chunk_number + 1]))
        print("Chunk:", chunk_number + 2, "New Data Size", len(working_train_set_data))
    end = time.time()
    print("Time:", end - start)


def evaluate_ensemble(device, test_loader, models, valid, chunk_number, criterion):
    ensemble = Ensemble(models).to(device)
    ensemble.eval()
    n_examples = 0
    n_correct = 0
    validation_loss = 0
    for examples, labels in (progress_bar := tqdm.tqdm(test_loader, desc="Ensemble Validation")):
        with torch.no_grad():
            examples = examples.to(device)
            labels = labels.to(device)
            logits = ensemble(examples)
            loss = criterion(logits, labels)
            n_correct += (logits.max(1).indices == labels).sum().item()
            validation_loss += loss.item()
            n_examples += labels.shape[0]

        progress_bar.set_postfix_str(
            f"Mean Loss: {validation_loss / n_examples:.5f}, Accuracy: {n_correct / n_examples:.4f}"
        )
    with open("ensemble_evaluation_results.csv", "a") as file:
        # chunk,n_retrained,loss,accuracy
        file.write(
            f"{chunk_number},{len(valid) - sum(valid)},{validation_loss / n_examples},{n_correct / n_examples}\n"
        )


def train_individual(
    model, optimizer, criterion, train_loader, test_loader, model_number, chunk_number, is_valid, epochs, device
):
    for epoch in (progress_bar := tqdm.tqdm(range(epochs), desc=f"Training model {model_number}")):
        model.train()
        for examples, labels in train_loader:
            examples = examples.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(examples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        validation_loss = 0
        n_correct = 0
        n_examples = 0
        for examples, labels in test_loader:
            examples = examples.to(device)
            labels = labels.to(device)
            logits = model(examples)
            loss = criterion(logits, labels)
            n_correct += (logits.max(1).indices == labels).sum().item()

            validation_loss += loss.item()
            n_examples += labels.shape[0]
        progress_bar.set_postfix_str(
            f"Mean Loss: {validation_loss / n_examples:.5f}, Accuracy: {n_correct / n_examples:.4f}"
        )

        with open(f"ensemble_training_results.csv", "a") as file:
            # model,chunk,valid,epoch,loss,accuracy
            file.write(
                f"{model_number},{chunk_number},{is_valid},{epoch},{validation_loss / n_examples},{n_correct / n_examples}\n"
            )


if __name__ == "__main__":
    main()
