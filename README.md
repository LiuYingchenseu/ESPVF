# Progressive Multi-Strategy Diagnosis for Early Stroke Recognition Using Facial and Upper Limb Movement Videos and EMR
### This project presents a Progressive Multi-Strategy Diagnosis Method (PMSDM) for early stroke recognition using facial and upper-limb movement videos with electronic medical records. The system integrates facial movement (CNN-LSTM), arm movement (accuracy and smoothness detection), and multimodal fusion models to assess stroke-related dysfunctions. Trained on real and simulated patient data, PMSDM achieved high accuracies (up to 94%).
## Requirements
- Python==3.9.18
- numpy==1.26.4
- scikit-learn==1.3.0
- pillow==10.4.0
## Project structure
```markdown
<pre>
├── ESPVF/                   
│   ├── face_project/            
|   |   ├──data/
|   |   ├──models/
|   |   ├──output/
|   |   ├──predict/
|   |   ├──utils/
|   |   ├──mian.py
│   ├── model_predict/            
|   ├── nose_project/
│   └── stroke_project/
└── README.md             
</pre>
```
## Training
```markdown
python model_predict.py
```
