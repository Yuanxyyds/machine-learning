# Machine Learning Projects Overview

Welcome to my collection of machine learning projects! Below is a detailed breakdown of two projects that I have developed. These projects focus on food classification using deep learning models and creating a personalized AI chatbot.

## 1. Food Classification with Food-101 Dataset

### Project Overview
This project focuses on building a deep learning-based food classification system using the [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101). The dataset contains 101,000 images of food items across 101 classes. However, I narrowed the classification task to 22 specific food categories. The system is trained using multiple deep learning models, including a Baseline Model, VGG Model, Inception Model, and ResNet Model, to classify images into one of the 22 categories.

The main goals of this project include:
- Data preparation and preprocessing of the Food-101 dataset.
- Applying data augmentation techniques to enhance the model's generalization.
- Integrating transfer learning to improve model performance with pre-trained weights.
- Hyperparameter tuning to optimize the models.
- Fine-tuning multiple models (Baseline, VGG, Inception, ResNet) to compare performance.

### Food Categories Supported:
The models classify food into the following 22 categories:
- Apple Pie
- Baby Back Ribs
- Bibimbap
- Caesar Salad
- Cheesecake
- Chicken Curry
- Chicken Wings
- Club Sandwich
- Donuts
- Dumplings
- French Fries
- Hot Dog
- Hamburger
- Frozen Yogurt
- Pizza
- Ramen
- Steak
- Ice Cream
- Waffles
- Spring Rolls
- Sushi
- Fish and Chips

### Key Features:
- **Data Augmentation**: Random transformations applied to images, such as rotation, zoom, flip, and shift to improve model generalization.
- **Transfer Learning**: Pre-trained VGG, Inception, and ResNet models were fine-tuned to speed up training and boost accuracy.
- **Hyperparameter Tuning**: Adjusting batch size, learning rate, and optimizers using Keras Tuner to find the best configurations.

### Models Used:
- **Baseline Model**: A simple Convolutional Neural Network (CNN) to set a baseline accuracy.
- **VGG Model**: Transfer learning with pre-trained weights from VGG16, fine-tuned on the 22 food classes.
- **Inception Model**: Transfer learning using InceptionV3, focusing on improving accuracy by leveraging its deeper architecture.
- **ResNet Model**: Fine-tuned ResNet architecture, known for its residual blocks, which helps with vanishing gradient issues in deeper networks.

The project is hosted on my home server, allowing efficient training using my local setup. I trained and fine-tuned the models on this server, optimizing resource usage.

---

## 2. StevenAI: Personalized AI Chatbot

### Project Overview
StevenAI is a personalized chatbot project designed to answer questions about me. The chatbot was built by fine-tuning the **LLaMA 3.2 model** with 3 billion parameters, specifically trained on approximately 1,000 Q&A pairs related to my background, experiences, and personal history. The chatbot can answer questions about my academic journey, hobbies, work experience, and more.

### Key Features:
- **Fine-Tuning with LoRA (Low-Rank Adaptation)**: To optimize the fine-tuning process, I used a LoRA adapter, which enables parameter-efficient fine-tuning. This approach significantly reduces the resource requirements for training large language models.
- **Unsloth Acceleration**: I incorporated the Unsloth library, which allows the model to train 2x faster and cuts down VRAM usage by 50%, making it more efficient to run on my **16GB NVIDIA RTX 4060Ti** GPU.
- **Hyperparameter Tuning**: The model was fine-tuned using various LoRA ranks (8, 16, 32, 64, 128) and LoRA alpha values. After testing different configurations, the best performance was achieved with a LoRA rank of 16 and LoRA alpha of 32.

### Technical Details:
- **Model**: LLaMA 3.2 (3 billion parameters)
- **Training Setup**: Fine-tuned using my home lab setup, which includes a 16GB NVIDIA RTX 4060Ti GPU. This setup allows for efficient training of large-scale models despite resource constraints.
- **Performance**: The model achieves a response accuracy of 70%-85% across different questions, though, due to the model size, occasional errors may occur.

### Limitations:
- **Occasional Mistakes**: While the chatbot performs well within the fine-tuned domain, it may sometimes provide inaccurate answers due to resource constraints and the complexity of the model.
  
Feel free to ask the chatbot questions about me, such as my academic background, hobbies, or work experience!

---

## Hardware and Environment Setup
Both projects were developed and trained within my home server environment, which includes:
- **CPU**: AMD Ryzen 5800X
- **GPU**: NVIDIA RTX 4060 Ti (16GB VRAM)
- **Server**: Running 24/7 for model training, fine-tuning, and hosting web services.
  
The food classification models and the StevenAI chatbot are hosted locally on this server for optimized resource usage.

---

## Credits and Acknowledgements
- **Food Classification**: Based on the work from [Kaggle's Food-101 Dataset](https://www.kaggle.com/datasets/kmader/food41) and inspired by [this notebook](https://www.kaggle.com/code/abdelrahmanahmed110/10-types-of-food-classification/notebook#Hyperparameter-Tuning-with-Keras-Tuner).
- **StevenAI**: Inspired by the fine-tuning techniques demonstrated in [this Colab Notebook](https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing).

---

## Future Plans
- **Food Classification**: Plan to extend the system to classify all 101 classes from the Food-101 dataset and further improve the models' accuracy through more advanced transfer learning techniques.
- **StevenAI**: Looking to expand the chatbot's knowledge base and improve its conversational capabilities by integrating new Q&A pairs and training with larger datasets.

Thank you for exploring my projects! Feel free to reach out with any questions or feedback. You can find more of my work on my [personal website](https://liustev6.ca).
