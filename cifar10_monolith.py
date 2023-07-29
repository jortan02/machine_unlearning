import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tqdm



def main():
    with open(f"monolith_training_results.csv", "w") as file:
        file.write("chunk,epoch,loss,accuracy\n")

    start = time.time()
    learning_rate = 0.001
    batch_size = 512
    chunks = 5
    subset_ratio = 0.2
    epochs = 50

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    working_train_set_data = train_set_chunks_data[0]
    working_train_set_targets = train_set_chunks_targets[0]

    model = torchvision.models.resnet152(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    print("Chunk:", 1)
    for chunk_number in range(chunks):
        training_size = len(working_train_set_data)
        sample_weights = torch.ones(training_size) / training_size
        sample_size = int(subset_ratio * training_size)
        train_sample_indices = sample_weights.multinomial(num_samples=sample_size, replacement=True)
        train_set_samples = data.TensorDataset(
            working_train_set_data[train_sample_indices], working_train_set_targets[train_sample_indices]
        )

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = torch.utils.data.DataLoader(
            train_set_samples, batch_size=batch_size, shuffle=True, num_workers=8
        )

        model.train()
        for epoch in (progress_bar := tqdm.tqdm(range(epochs), desc=f"Training monolith chunk {chunk_number + 1}")):
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
            with open(f"monolith_training_results.csv", "a") as file:
                file.write(f"{chunk_number + 1},{epoch},{validation_loss / n_examples},{n_correct / n_examples}\n")

        if chunk_number + 1 < chunks:
            working_train_set_data = torch.cat((working_train_set_data, train_set_chunks_data[chunk_number + 1]))
            working_train_set_targets = torch.cat((working_train_set_targets, train_set_chunks_targets[chunk_number + 1]))

        print("Chunk:", chunk_number + 2, "New Data Size", len(working_train_set_data))
    end = time.time()
    print("Time:", end - start)


if __name__ == "__main__":
    main()
