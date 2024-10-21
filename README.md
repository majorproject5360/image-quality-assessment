## Image Quality Assessment System using Streamlit and CNN

This project provides an image quality comparison system with a frontend built using Streamlit. The system enables users to upload images, analyze their quality using a pre-trained CNN model (saved as `model.keras`), and recommend the best quality image.

---

### Features
- Upload multiple images through the web interface.
- Analyze the quality of uploaded images.
- Display metrics for quality assessment.
- Recommend the best quality image.
- Simple and intuitive UI using Streamlit.

---

### Setup and Installation

1. Clone the Repository


```bash
git clone https://github.com/majorproject5360/image-quality-assessment.git
cd major-project/streamlit-website/image-quality-assessment
```
ml_model.py file contains python code which needs to be converted to .keras format, it can be achieved by running below code mentioned at end of the file:

# Save the model in .keras format
model.save('/content/model.keras')

from google.colab import files
files.download('/content/model.keras')

NOTE: Run ml_model.py file in Google Colab and above mentioned snippet of code will save model as model.keras and download it to your local system. Add model.keras file to the project directory.

2. Install Dependencies
Make sure Python 3.8+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit App
```bash
streamlit run app.py
```
This will start a local server. Visit [http://localhost:8501](http://localhost:8501) in your browser to access the web interface.

---

### How It Works
1. Upload one or more images through the web interface.
2. The app analyzes the images using the CNN model.
3. Quality scores are displayed, and the system suggests the best image.

---

### Technologies Used
- Python: Core language
- TensorFlow: CNN model loading and predictions
- Streamlit: Frontend web interface
- Pillow: Image processing
- scikit-learn: ML utilities

---

### Future Improvements
- Support for more image quality metrics.
- Option to fine-tune the model using new datasets.
- Integration with AWS for model deployment.

---

### Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

### Acknowledgements
- Special thanks to the Streamlit community for providing a simple yet powerful UI framework.
- CNN model architecture inspired by tutorials on image quality assessment.

---
