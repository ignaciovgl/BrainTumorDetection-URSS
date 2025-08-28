import gradio as gr
import torch
import torch.nn as nn
import cv2
import numpy as np
from pyro.nn import PyroModule, PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoMultivariateNormal
import pyro.distributions as dist
from torchvision.models import resnet18, ResNet18_Weights
import torch.serialization
import torch.distributions.constraints as constraints
import pyro.distributions.constraints as pyro_constraints
import pyro.poutine as poutine

# 1. Define your model architecture. This is a copy from your notebook.
class BayesianResNet18(PyroModule):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        to_pyro_module_(self.resnet)
        
        self.resnet.fc = PyroModule[nn.Linear](num_ftrs, num_classes)
        self.resnet.fc.weight = PyroSample(dist.Normal(torch.tensor(0., device="cuda"), torch.tensor(1., device="cuda")).expand([num_classes, num_ftrs]).to_event(2))
        self.resnet.fc.bias = PyroSample(dist.Normal(torch.tensor(0., device="cuda"), torch.tensor(10., device="cuda")).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = self.resnet(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=x), obs=y)
        return obs

# 2. Instantiate the model and load your saved states.
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnn_model = BayesianResNet18(num_classes).to(device)
guide = AutoMultivariateNormal(poutine.block(bnn_model, hide=['obs']))

saved_state = torch.load('bnn_model_and_guide.pth', map_location=device,weights_only=False)

# Load the model's state_dict
bnn_model.load_state_dict(saved_state['model_state_dict'])

# Load the guide's parameters using the Pyro parameter store
pyro.get_param_store().set_state(saved_state['guide_params'])

bnn_model.eval()
guide.eval()

# 3. Create a prediction function for Gradio.
# This function will be called every time a user uploads an image.
def predict(input_image):
    # Preprocess the input image to match the model's training data.
    # The image is a PIL image from Gradio, convert it to numpy array.
    image_np = np.array(input_image)
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=-1)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)

    # Replicate the single grayscale channel to 3 channels for ResNet18.
    image_tensor = image_tensor.repeat(3, 1, 1).unsqueeze(0).to(device)

    # Use Pyro's Predictive to get multiple samples for uncertainty estimation.
    predictive = Predictive(bnn_model, guide=guide, num_samples=100)
    with torch.no_grad():
        predictions = predictive(image_tensor)

    posterior_samples = predictions['obs']
    
    # Calculate the distribution of predictions to get probabilities.
    sample_counts = torch.bincount(posterior_samples.squeeze(), minlength=num_classes)
    total_samples = posterior_samples.shape[0]
    probabilities = (sample_counts.float() / total_samples).cpu().numpy()

    # Get the most likely class (mode of the samples).
    mode_prediction = torch.mode(posterior_samples.squeeze(), dim=0).values.item()
    predicted_label = class_names[mode_prediction]

    # Convert probabilities to a dictionary format that Gradio's Label component can use.
    output_probabilities = {class_names[i]: float(probabilities[i]) for i in range(num_classes)}
    
    return predicted_label, output_probabilities

# 4. Define the Gradio interface.
# The interface will have an image input and two outputs: a label and a chart.
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(label="Predicted Tumor Type"), gr.Label(label="Uncertainty Estimation")],
    title="Brain Tumor Classification with Uncertainty Estimation",
    description="Upload a brain MRI scan to get a tumor type prediction and an estimation of the model's certainty."
)

# Launch the app.
iface.launch(debug=True)