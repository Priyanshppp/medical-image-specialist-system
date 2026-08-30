\# Medical Image Specialist System



A modular medical image classification system using modality-specific

specialists and candidate-constrained prediction.



\## Project Overview



This project explores a specialist-based approach for medical image

classification and multiple-choice visual question answering.



Instead of using one classifier for every medical image type, separate

specialists are trained for different imaging modalities.



\## Supported Modalities



\- CT

\- Dermatology

\- Fundus

\- MRI

\- Microscopy

\- OCT

\- Ultrasound

\- X-Ray



\## Pipeline



Image

→ Feature Extraction

→ Modality Specialist

→ Candidate-Constrained Prediction

→ Final Answer



\## Feature Extraction



Pretrained ResNet18 is used to extract 512-dimensional image feature

vectors.



Training and validation data are processed separately.



\## Specialist Architecture



A separate classifier is trained for each imaging modality.



Current specialists:



\- CT Specialist

\- Dermatology Specialist

\- Fundus Specialist

\- MRI Specialist

\- Microscopy Specialist

\- OCT Specialist

\- Ultrasound Specialist

\- X-Ray Specialist



\## Candidate-Constrained Prediction



Instead of allowing a classifier to predict any class, predictions are

restricted to the answer choices provided in each question.



The candidate with the highest compatible classifier score is selected.



\## Initial Results



| Modality | Raw Accuracy | Candidate-Constrained Accuracy |

|----------|-------------|--------------------------------|

| Ultrasound | 30% | 80% |

| CT | 25% | 75% |



\## Current Progress



Completed:



\- Project structure creation

\- Training/validation manifest preparation

\- Modality-specific manifests

\- ResNet18 feature extraction

\- Eight modality specialist models

\- Ultrasound candidate-constrained evaluation

\- CT candidate-constrained evaluation

\- Generalized specialist evaluation framework



\## Next Steps



\- Evaluate remaining modality specialists

\- Improve candidate matching

\- Build modality routing

\- Build unified inference pipeline

\- Final validation

\- Test inference and submission generation



\## Important Experimental Constraint



Final test data is not used for specialist training or validation.

