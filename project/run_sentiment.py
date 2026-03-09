import random
import itertools

import embeddings

import minitorch
from datasets import load_dataset

BACKEND = minitorch.TensorBackend(minitorch.FastOps)


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Cat(minitorch.Function):
    @staticmethod
    def forward(ctx, dim_tensor: minitorch.Tensor, *tensors: minitorch.Tensor) -> minitorch.Tensor:
        ctx.save_for_backward(dim_tensor, [t.shape for t in tensors])
        dim = int(dim_tensor.item())
        
        # Validate input shapes
        shape = list(tensors[0].shape)
        for t in tensors[1:]:
            if len(t.shape) != len(shape):
                raise ValueError("All tensors must have the same number of dimensions")
            for i, s in enumerate(t.shape):
                if i != dim and s != shape[i]:
                    raise ValueError(f"Sizes of tensors must match except in dimension {dim}")
        
        # Calculate output shape
        total_dim = sum(t.shape[dim] for t in tensors)
        out_shape = list(shape)
        out_shape[dim] = total_dim
        
        # Create output tensor
        out = minitorch.zeros(tuple(out_shape), backend=tensors[0].backend)
        
        current_offset = 0
        for t in tensors:
            # Iterate over all elements of t
            ranges = [range(s) for s in t.shape]
            for index in itertools.product(*ranges):
                val = t[index]
                
                # Calculate target index
                target_index = list(index)
                target_index[dim] += current_offset
                
                out[tuple(target_index)] = val
            
            current_offset += t.shape[dim]
            
        return out

    @staticmethod
    def backward(ctx, grad_output: minitorch.Tensor):
        dim_tensor, shapes = ctx.saved_values
        dim = int(dim_tensor.item())
        
        grads = []
        current_offset = 0
        
        for shape in shapes:
            # Create gradient tensor for this input
            grad = minitorch.zeros(shape, backend=grad_output.backend)
            
            # Copy from grad_output
            ranges = [range(s) for s in shape]
            for index in itertools.product(*ranges):
                # Source index in grad_output
                source_index = list(index)
                source_index[dim] += current_offset
                
                val = grad_output[tuple(source_index)]
                grad[tuple(index)] = val
                
            grads.append(grad)
            current_offset += shape[dim]
            
        return (minitorch.zeros((1,), backend=grad_output.backend),) + tuple(grads)


def cat(tensors, dim=0):
    if not tensors:
        return minitorch.zeros((0,), backend=BACKEND)
    dim_tensor = minitorch.tensor([dim], backend=tensors[0].backend)
    return Cat.apply(dim_tensor, *tensors)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        """
        Apply 1D convolution to the input tensor.

        Args:
            input: Input tensor of shape (batch, in_channels, width)

        Returns:
            Output tensor of shape (batch, out_channels, new_width)
            where new_width = width - kernel_width + 1
        """
        # Apply 1D convolution: convolve input with learned kernel weights
        conv_out = minitorch.conv1d(input, self.weights.value)
        # Add bias term (broadcast across batch and width dimensions)
        return conv_out + self.bias.value


class CNNSentimentKim(minitorch.Module):
    """
    Implement a CNN for Sentiment classification based on Y. Kim 2014.

    This model should implement the following procedure:

    1. Apply a 1d convolution with input_channels=embedding_dim
        feature_map_size=100 output channels and [3, 4, 5]-sized kernels
        followed by a non-linear activation function (the paper uses tanh, we apply a ReLu)
    2. Apply max-over-time across each feature map
    3. Apply a Linear to size C (number of classes) followed by a ReLU and Dropout with rate 25%
    4. Apply a sigmoid over the class dimension.
    """

    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_filters = len(filter_sizes)

        # Build conv layers using helper method
        self.convs = self._build_conv_layers(embedding_size, feature_map_size, filter_sizes)

        # Output layer
        self.linear = Linear(feature_map_size * self.num_filters, 1)

    def _build_conv_layers(self, in_channels, out_channels, kernel_sizes):
        """Create and register Conv1d layers with different kernel sizes."""
        convs = []
        for i, k in enumerate(kernel_sizes):
            conv = Conv1d(in_channels, out_channels, k)
            setattr(self, f"conv_{i}", conv)
            convs.append(conv)
        return convs

    def _apply_conv_block(self, x, conv):
        """Apply single conv block: conv -> relu -> max-pool -> flatten."""
        out = conv(x).relu()
        out = minitorch.max(out, dim=2)
        return out.view(out.shape[0], out.shape[1])

    def _to_channels_first(self, embeddings):
        """Convert [batch, seq_len, channels] to [batch, channels, seq_len]."""
        return embeddings.permute(0, 2, 1)

    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, sentence_length, embedding_dim]
        Returns:
            Probability scores [batch]
        """
        x = self._to_channels_first(embeddings)

        # Apply all conv blocks and concatenate
        conv_outputs = [self._apply_conv_block(x, conv) for conv in self.convs]
        x = cat(conv_outputs, dim=1)

        # Classification head
        x = minitorch.dropout(x, self.dropout, ignore=not self.training)
        x = self.linear(x)
        return x.sigmoid().view(x.shape[0])


# Evaluation helper methods
def get_predictions_array(y_true, model_output):
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array):
    correct = 0
    for y_true, y_pred, logit in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    val_acc = validation_accuracy[-1] if len(validation_accuracy) > 0 else 0.0
    best_val = max(best_val, val_acc)
    
    log = (
        f"Epoch {epoch}, loss {train_loss:.5f}, "
        f"train accuracy {train_accuracy[-1]:.2%}, "
        f"val accuracy {val_acc:.2%}, "
        f"best val {best_val:.2%}"
    )
    print(log)
    with open("sentiment.txt", "a") as f:
        f.write(log + "\n")


class SentenceSentimentTrain:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                # Save train predictions
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]

                # Update
                optim.step()

            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0


def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        # TODO: move padding to training code
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    #  Determine max sentence length for padding
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    with open("sentiment.txt", "w") as f:
        f.write("")  # Clear file
    
    train_size = 450
    validation_size = 100
    learning_rate = 0.05
    max_epochs = 15

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )