using Flux
using Images
using ImageIO

model = Chain(
    Conv((3, 3), 1 => 16, relu),   # First convolutional layer
    MaxPool((2, 2)),                # Max pooling layer
    Conv((3, 3), 16 => 32, relu),   # Second convolutional layer
    MaxPool((2, 2)),                # Another max pooling layer
    Flux.flatten,                   # Flatten the output for fully connected layer
    Dense(32 * 5 * 5, 10)           # Dense layer (assuming the input image is reduced to 5x5 after pooling)
)


img = load("generated_image_2 (2).png")
img = Gray.(img)
img = imresize(img, (28, 28))

img_array = Float32.(img)  # Convert to Float32 array
img_tensor = reshape(img_array, (28, 28, 1, 1))  # Reshape to match model input


output = model(img_tensor)  # Forward pass


output_probabilities = softmax(output)


println("Model Output (Probabilities): ", output_probabilities)
