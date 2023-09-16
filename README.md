PHASE-4-PROJECT
Link to presentation [slides](https://docs.google.com/presentation/d/1oCcRw6qBNtLyPmFcWuvbvYy4MqiaZIWUIpUATGp9adA/edit#slide=id.g1f87997393_0_1466)

# A PNEUMONIA CHEST X-RAY IMAGE CLASSIFICATION FOR JMARKS HOSPITAL

***
## Introduction
With change in time and technology, healthcare has drastically changed over the time. Statistics show that there has been a significant increase in the world population and high infection rates. Doctors and Radiologists at JMarks Hospital have been affected by these changes hence compromising the level of efficiency of healthcare delivery in the hospital. Screening of thousands of patients for chest x-rays to determine the diagnosis for pneumonia has incresingly become tiresome and costly. Moreover, interpretation of chest x-ray images by doctors is limited to accuracy since its a judgement made by naked human eyes. The hospital administration is therefore seeking to mitigate this matter by building a sysytem or tool that can assist doctors and radiologists in diagnosing a patient for pheumonia, but the lack knowledge on how to do that.

* N\B: One of the key questions that arises concerns the ability of a system to correctly work in order to ensure that the quality of healthcare being delivered meets the expectations of the doctors, radiologists and the patients receiving services.

This question is at the heart of this project, where I delve into the realm of classification analysis to uncover insights that can shape the future of the healthcare sector. The focus is on JMarkss Hospital which dedicated in providing cutting-edge services to its patients. By harnessing the power of Image Classification analysis and machine learning, I aim to provide JMarkss Hospital and its stakeholders with the tools to correctly identify a patient with pneumonia and implement targeted efforts for retaining their valuable customer base. We also need to understand what pneumonia is:

#### What is Pneumonia?
Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.
***

### Business Understanding:
***
This project aims to develop a deep learning model that can analyze chest X-ray images and classify whether a patient has pneumonia. This classification can help doctors and healthcare providers at JMarkss Hospital identify cases of pneumonia more quickly and accurately, allowing for prompt treatment and improving patient outcomes.

### Stakeholders:
The success of this project will be of great use to different JMarkss Hospital stakeholders:

* JMarkss Hospital Management: As the ultimate decision-makers, they're vested in the project's outcomes for improved healthcare delivery.
* Doctors, Radiologists and healthcare professionals: They are the primary users of this model in their daily routine.
* Data Science Team (Project Team): Responsible for executing the project, analyzing data, and creating classification models.

### Direct Impact:
The execution of this project directly affects the core operations of JMarkss Hospital, impacting quality of healthcare delivery, revenue streams, and customer satisfaction levels.

### Business Problem(s) Solved:
This Data Science endeavor addresses the critical business problem of pneumonia chest x-ray detection. It aims to identify patients with pnemonia and those without to improve efficiency and quality of healthcare system. In this sense, the research questions are:

   * Is the model correcly working?

   * What is the accuracy level of the model?
    
   * How much time and resources is reduced by using this model?

### Scope of the Project:
Inside the project's scope are the following components:

* Pneumonia Prediction: Developing an Image classification models to detect a sick patient and a healthy one.
* Correct Classification: Correctly classifying chest x-ray images.
* Recommendations: Offering actionable suggestions to curb pneumonia problem in the hospital and the community at large.

### Outside the Scope:
While the project tackles the formidable challenge of image classification, certain aspects lie beyond its immediate purview. Such as, the implementation of recommended strategies to running the model and the evaluation of the financial impact arising from the project's outcomes is a distinct consideration.
***

### Problem Statement
***
In the landscape of modern healthcare sector, the challenge of a huge number patients and rise of disastrous diseases require better systems. This project aims to enhance healthcare delivery by providing a model that can correctly classify chest x-ray images therefore improving on efficiency, effectiveness, timeliness, accuracy and quality of healthcare delivery to the delicate customers.

### Challenge:
The primary challenge lies in the correct classification of images since a patients wellbeing is a life sensitive matter which is import to the stakeholders as well.

### Objective:
The objective of this project is to build a classifier developing accurate predictive models capable of identifying pneumonia conditions from a chest x-ray image.

### Benefits
By successfully addressing the challenge of correctly classifying chest x-ray images, this project stands to gain several benefits:
* Enhanced Patient Satisfuction: by the patients and doctors.
* Optimized Resource Allocation: this will enable optimizing operational efficiency.
* Business Sustainability: the hospital's qualty of care is improved and its reputation built which a solid foundation for long-term growth.
***

### Conclusion:
The project's implications are significant, as it has the potential to assist healthcare professionals in diagnosing pneumonia more effectively, which can ultimately save lives and reduce healthcare costs.

***
## Data Understanding:
### Data Sources:
The dataset is sourced from Kaggle, [document here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) which provides a large collection of chest X-ray images labeled with pneumonia or non-pneumonia. This dataset is suitable for the project as it contains real-world medical images necessary for training a pneumonia classification model.

This project encompasses, machine learning model development, and the interpretation of model results. It involves understanding inner workings of the model to make correct predictions and make sure that the model works.

### Dataset Size:
The dataset consists of thousands of chest X-ray images, with labels indicating whether each image contains pneumonia or not.

* Descriptive Statistics:
Descriptive statistics will be provided for the pixel values of the images, but no traditional statistical analysis is required for image data.

* Feature Justification:
The pixel values of the chest X-ray images are the relevant features for this project. These images capture important visual information for pneumonia diagnosis.

* Data Limitations:
Class Imbalance: The dataset may have class imbalance issues, with more non-pneumonia cases than pneumonia cases. This can affect model training and evaluation.
Data Quality: The quality of the X-ray images, variations in image resolution, and potential noise can impact model performance.


Data augmentation to increase the diversity of the training dataset.
Splitting the dataset into training, validation, and test sets.
Preprocessing the images, including resizing and normalization.
The steps taken are appropriate as they ensure that the dataset is ready for training and evaluation of deep learning models.
***

## Data Preparation
Importing of relevant packages for use in the analysis proccess, loading and organizing the dataset, explore the data and preprocess it to ensure proper evaluation of deep learning models.

## Loading the datasets 
### Description of the Pneumonia Dataset
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Data Preprocessing and Data Augmentation
In order to avoid overfitting problem, we need to expand artificially our dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations. Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. By applying these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.

For the data augmentation, i choosed to :

1. Randomly rotate some training images by 30 degrees
2. Randomly Zoom by 20% some training images
3. Randomly shift images horizontally by 10% of the width
4. Randomly shift images vertically by 10% of the height
5. Randomly flip images horizontally. Once our model is ready, we fit the training dataset.

Then we downsample the data in the train to have 1000 images for training. Downsalmping saves the model from running for longer periods hence speeds up the training process.

# Modeling:
The modeling phase will involve an iterative approach, starting with a simple baseline model and then introducing more complex models such as Convolutional Neural Networks (CNNs). The steps will include:

Building a baseline model using a densely connected neural network.
Introducing CNNs to capture spatial information effectively.
Visualizing intermediate activations to gain insights into the model's decision-making process.
Applying regularization techniques to mitigate overfitting.
Model changes will be justified based on their impact on model performance and relevance to the problem.
## The Baseline model (A Densely Connected Network)

### Explore the data again
Let's visualize some sample images to see the images in the two classess, pneumonia and normal chest x-rays.
Build a baseline fully connected model.
### Interpreting baseline model results
The fitted baseline model has an accuracy level of about 74 percent from ten epochs on the training data. This means that it can overally make correct predictons of about 74 percent on the x-ray images. The loss figures were relatively high with about 59 percent. We should also note that a learning rate was used here.
The baseline model has recorded an accuracy level of about 62 percent on the testing data while the loss is slightly higher with a figure of about 67 percent. We can introduce a second model, the convolutuion neural network to see how the results behave.

## Convolution Neural Network Model

For the second model we are using a convolution neural network model which is better since it identifies patterns in images which is great for image clssification.
The second model, convolution neural network, shows an improved result with about 89 percent which is an improvement from the baseline model. This shows that the CNN model is better than the baseline model and can be deployed for prediction and classification of the problem at hand.

## Analysis after model training

## Evaluation
The evaluation phase will assess the model's performance in solving the problem of pneumonia classification for JMarkss Hospital. Key aspects include:

- Choice of Metrics.
- Final Model Selection: The model that performs best on the chosen metrics with validation data will be selected.
- Discussion of Implications: The implications of the final model evaluation will be discussed in the context of solving the pneumonia classification problem.
The model has an average accuracy of about 87 percent on the testing data. This is a fair figure indicating that the model can overally works and can make predictions. The model also recorded a test loss of about 36 percent which is a fair figure. Based on these results we can conclude that the model can be used by the stakehoders for solving the problem but with close monitoring. Let's see how it makes predictions on the data.

## Make prediction
### Interpreting prediction results
The model is performing on average as from the classification report which shows an accuracy level of about 54 percent on the test data and a weighted average of about 48 percent in the test set.

The precision, recall and f1-score are generally low explaining about 35 percent of the data, although it is important to note that the model has performed better on other metrics such as accuracy score.

***
#### Conclusions:

- Model Performance: In general, the model is performing on average and is able to make predictions. The convolution neural network as the final model showed an improved performance and accuracy on the training and testing datasets.
- Usability of the model: The model is able to make correctpredictions based on the evaluation report. The healthcare professionals using this model should continue keep track of its performance and provide continous feedback

#### Recommendations:
- Model Selection: Based on the model evaluations, we recommend deploying convolution neural network model which showed impressive results on the metrics.
- Data Collection: To further enhance the models, consider collecting additional data that might have predictive power. This could include more detailed x-ray images and other behavioral attributes.
- Regular Updates: The CNN prediction model should be updated regularly with new data. This ensures that the model remain relevant and effective in capturing changing behavior and preferences.
- Feedback Loop: Continuously monitor the performance of deployed models and gather feedback from business stakeholders. This feedback loop can help refine the models and identify areas for improvement.
- Interpretability: While more complex models can yield better performance, it's important to balance this with model interpretability. Explainable models can provide insights into the reasons behind predictions, aiding decision-making.

In conclusion, deploying the predictive models for chest x-ray image classification can significantly benefit JMarkss Hospital by enabling them to correcly identify patients with pneumonia condition. The model should be regularly monitored, updated, and integrated into business processes for effective decision-making and improved healthcare delivery process. 

### Next Steps: 
To improving the model, such as 
1. Collecting more diverse data.
2. Highlighting ethical considerations, privacy concerns, and regulatory compliance in implementing the model.
3. Exploring transfer learning from pre-trained models.