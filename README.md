# Waste Sorting Using Computer Vision  

## **Introduction**  
Waste management is crucial for sustainability, and improper waste sorting reduces recycling efficiency. This project automates the waste sorting process using a convolutional neural network (CNN) model with transfer learning.  

## **Objective**  
To classify waste into six categories: plastic, glass, paper, metal, shoes, and clothes, leveraging a pre-trained ResNet50 model.  

---

## **Dataset**  
- **Source:** The dataset is taken from [Kaggle](https://www.kaggle.com).  
- **Structure:** Six categories with approximately 700–1000 images per class.  
- **Data Splitting:**  
  - Training: 70%  
  - Validation: 15%  
  - Testing: 15%  

---

## **Model Architecture**  
### **Base Model:** ResNet50  
- Pre-trained on ImageNet.  
- Fine-tuned for better classification.  

### **Classification Head:**  
- Global Average Pooling Layer  
- Dense Layer (512 units, ReLU activation)  
- Output Layer (6 units, Softmax activation)  

---

## **Training Details**  
- **Initial Training:**  
  - Optimizer: Adam (learning rate = 0.001)  
  - Loss: Categorical Crossentropy  
  - Epochs: 10  

- **Fine-Tuning:**  
  - Unfrozen the last four layers of ResNet50.  
  - Optimizer: Adam (learning rate = 1e-5)  
  - Epochs: 5  

---

## **Performance**  
- **Test Accuracy:** 67.43%  
- **Metrics:** Precision, Recall, and F1-Score for all classes.  
- **Confusion Matrix:** Highlighted areas for improvement in distinguishing between similar categories.  

---

## **Future Work**  
1. **Data Collection:** Expand the dataset to reduce class imbalance.  
2. **Model Enhancement:** Experiment with architectures like EfficientNet or InceptionNet.  
3. **Hyperparameter Tuning:** Optimize for better performance.  
4. **Real-World Testing:** Deploy in recycling facilities.  
5. **Preprocessing Improvements:** Use techniques like segmentation or edge detection.  

---

## **How to Run the Project**  

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/XabkaBaap/waste-sorting-Using_Computer-Vision.git
2. ```bash
   pip install -r requirements.txt
