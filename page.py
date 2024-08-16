from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO
from torchvision import models
from torch import nn

# Initialize FastAPI app
app = FastAPI()

# Define the VGG model architecture
def create_vgg_model():
    model = models.vgg16(weights='DEFAULT')  # Use the latest weights
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model

# Load the trained model
model_path = 'best_model.pth'  # Update with the correct path if needed
model = create_vgg_model()  # Initialize the model structure
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# HTML form to upload an image
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Brain Tumor Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f4f4f9;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        form {
            margin-top: 20px;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 15px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>Brain Tumor Detection</h1>
    <p>Upload an image to check for a tumor.</p>
    <form action="/predict/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Upload Image">
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and transform the image
    image = Image.open(BytesIO(await file.read())).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    # Return HTML response with prediction result
    result = "Tumor Present" if prediction == 1 else "No Tumor"
    result_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f4f4f9;
                padding: 20px;
            }}
            h2 {{
                color: #333;
            }}
            .result {{
                margin-top: 20px;
                font-size: 20px;
                color: #555;
            }}
            a {{
                text-decoration: none;
                color: #4CAF50;
                font-size: 18px;
            }}
            a:hover {{
                color: #45a049;
            }}
        </style>
    </head>
    <body>
        <h2>Prediction Result:</h2>
        <p class="result">{result}</p>
        <a href="/">Upload another image</a>
    </body>
    </html>
    """
    return HTMLResponse(content=result_html)
