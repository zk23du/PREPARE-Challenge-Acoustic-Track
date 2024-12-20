## **PREPARE Challenge - Acoustic Track**

---

### **1. Objective**
The objective of this project is to develop an efficient and accurate system to classify audio files into three categories:
- **diagnosis_control**
- **diagnosis_mci**
- **diagnosis_adrd**

The system aims to extract meaningful embeddings from raw audio files, train a classification model, and achieve high performance on the test set.

---

### **2. Introduction**
Alzheimer's disease and Alzheimer's disease-related dementias (AD/ADRD) are a set of brain disorders affecting more than 6 million Americans. Early intervention is crucial for successful disease modification, but detecting early signs of cognitive decline and AD/ADRD remains challenging. Current clinical methods often lack the sensitivity needed for early prediction, especially in underrepresented groups.

The aim of this challenge track is to improve early prediction of Alzheimer's disease and related dementias (AD/ADRD) using acoustic biomarkers from voice recordings. Through this initiative, the National Institute on Aging (NIA) aims to improve accuracy across diverse populations and explore understudied factors that may indicate early AD/ADRD.

---

### **3. Project Structure**
```
├── audios
│   ├── whisper_best
│   │   └── main.py
│   └── (Other audio processing and prediction files)
├── csv
│   └── (Code files for processing CSV files provided in the challenge)
├── README.md
└── (Other necessary files and folders)
```

---

### **4. Key Components**

#### **1. ASR Models and Embedding Extractors**
We have utilized **ASR (Automatic Speech Recognition) models** and other embedding extraction techniques to extract meaningful representations from the raw audio files. The key components are as follows:
- **Embedding Extractors**: Multiple embedding extractors are supported and can be selected within the `main.py` file.
- **Models**: Different models are available to train on the extracted embeddings, and you can select the desired model when running `main.py`.

#### **2. Audio Processing and Prediction**
- All the code related to **audio processing** and **prediction** is located in the `audios` folder.
- The **best-performing script** is located at: `audios/whisper_best/main.py`.
- To run the code, you can use the following command:
  ```bash
  python3 main.py
  ```
  While running `main.py`, you can select the desired **embedding extractor** and **model** as required for experimentation.

#### **3. CSV Processing**
- The `csv` folder contains scripts for **processing CSV files** provided for the challenge.
- This code is used to clean, preprocess, and organize the metadata or labels associated with the audio files.

---

### **5. Usage Instructions**
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd prepare-challenge
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Script**:
   - Navigate to the `audios/whisper_best` folder.
   - Run the script with the following command:
     ```bash
     python3 main.py
     ```
   - You can specify the **embedding extractor** and **model** of your choice in the script to customize the process.

---

### **6. How to Customize**
- **Change Embedding Extractor**: Open the `main.py` file and change the variable related to the embedding extractor to select from multiple options.
- **Change Model**: The model to be used for classification can be changed by updating the model selection logic within the `main.py` file.

---

### **7. Best Model**
- The best-performing model in our implementation is found in the **whisper_best** folder.
- The code for this model can be run as described in the **Usage Instructions** section.
- You can modify the **embedding extractor** and **model** directly within `main.py` to try different configurations.

---

### **8. Contact**
If you have any questions or suggestions about the project, feel free to create an issue in the repository or contact the project maintainers.

---

### **9. Acknowledgments**
We acknowledge the efforts of the **National Institute on Aging (NIA)** for initiating the PREPARE Challenge and promoting research in Alzheimer's disease and related dementias (AD/ADRD).

