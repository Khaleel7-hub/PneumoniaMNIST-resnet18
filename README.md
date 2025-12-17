# PneumoniaMNIST-resnet18
Final Project Instructions: Utilizing and Explaining CNN Architectures for Medical Image Classification

# **Objective:**

Use Convolutional Neural Network (CNN) architectures to classify medical images, specifically using the PneumoniaMNIST dataset. Document your process and findings in a detailed report.

**Specific Requirements:**

* **Dataset**: Use the PneumoniaMNIST dataset.
* **Preprocessing**: There is no need for additional preprocessing as the dataset is already processed.
* **Code and Resources**: All required code for loading, training, and evaluating the model is available on the MedMNIST GitHub Repository. Refer to the following resources:
  * [Getting Started Notebook](https://colab.research.google.com/github/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb) for a toy example.
  * [Experiments Repository](https://github.com/MedMNIST/experiments) for full-fledged training of ResNet18 and ResNet50. Use ResNet18 for this project.

* **Environment**: Use Google Colab for implementation. Ensure to mount your Google Drive using:
```
    from google.colab import drive
    drive.mount('/gdrive')
```
And change the runtime to GPU for efficient training.
# Tasks:
1. **Data Preparation**:
Use the **PneumoniaMNIST** dataset, which requires no additional preprocessing.
2. **Model Training**:
Implement and train a ResNet18 model using the provided resources and examples. You can choose to either train from scratch or use transfer learning with preloaded weights. 
* If you choose training from scratch, you should try to experiment with different learning rates to try to improve your result.
* If you choose transfer learning, you should try to experiment with freezing different layers. 
3. **Model Evaluation**
Evaluate the model's performance using the test set.
Generate and analyze a confusion matrix and ROC curve.
4. **Explainability with SHAP**
* Apply SHAP for global and local explanations of the model.
* Analyze at least two test cases with correct and incorrect predictions, from different classes.
* Use Gradient Explainer for image classification. You can find examples on how to apply it in the [official shap repo](https://github.com/shap/shap/tree/master/notebooks/image_examples)
5. **Report Writing**
* Document each step, from data preparation to explainability.
* Include figures for model evaluation and SHAP analysis, and discuss their significance.
* Address any challenges and insights.

# Deliverables:
* Code implementation in Google Colab.
* Trained ResNet18 model.
* Evaluation and explanation plots.
* Comprehensive final report.

* A presentation will be held 19/12.
