# Text-Classification-of-AEC-domain

This repository contains the official code for the paper:

**A Multi-model Collaborative Text Classification Approach for the Architecture, Engineering and Construction Domain**

---

## 🛠️ Environment

- Python 3.8
- torch==2.6.0
- numpy==2.3.1
- InstructorEmbedding==1.0.1
- sentence_transformers==2.2.2
- openai==0.28.1
- scikit_learn==1.3.0

---

## 📂 Datasets

We utilize two publicly available datasets in Chinese and English:

- **OSHA Accident Reports (English)**  
  [https://github.com/safetyhub/OSHA_Acc/blob/master/tagged1000.xlsx](https://github.com/safetyhub/OSHA_Acc/blob/master/tagged1000.xlsx)

- **Building Code Dataset (Chinese)**  
  [https://github.com/SkydustZ/Text-Classification-Based-Approach-for-Evaluating-and-Enhancing-Machine-Interpretability-of-Building/tree/main/CivilRules/dataset](https://github.com/SkydustZ/Text-Classification-Based-Approach-for-Evaluating-and-Enhancing-Machine-Interpretability-of-Building/tree/main/CivilRules/dataset)

---

## 🔗 Pretrained Models and Train Results

All pretrained models and train results can be downloaded here:  
📎 [https://pan.baidu.com/s/1u6SZPqk5dWVUQglbwlZa7Q?pwd=w49u](https://pan.baidu.com/s/1u6SZPqk5dWVUQglbwlZa7Q?pwd=w49u)

Please place them in the following directories:

- `building_code/saved_model/` and `building_code/pretrained_models/`
- `OSHA/best_model.pt`
- `OSHA/test_bert_roberta/best_bert.pth`  
  `OSHA/test_bert_roberta/best_roberta.pth`

---

## 🧪 Reproducing the Results

### 🔹 OSHA Dataset

- **BERT & RoBERTa fine-tuning**  
  ```bash
  cd OSHA/test_bert_roberta
  python test.py
  ```

- **Instructor model**  
  ```bash
  cd OSHA
  python test.py --model instructor
  # When prompted to generate prediction results, input "y"
  ```

- **Multi-model collaboration with Qwen**  
  ```bash
  python test.py --model qwen
  ```

- **Multi-model collaboration with DeepSeek**  
  ```bash
  python test.py --model deepseek
  ```

### 🔹 Building Code Dataset

- **BERT or RoBERTa fine-tuning**  
  ```bash
  python test.py --model bert
  python test.py --model roberta
  ```

- **Multi-model collaboration with LLaMA/Qwen/DeepSeek**  
  ```bash
  python test.py --model s_s_b
  # Then input model name (llama / qwen / deepseek)
  ```

- **Single DeepSeek model prediction**  
  ```bash
  python test.py --model deepseek_only
  ```

- **Fine-tuned LLaMA or Qwen prediction**  
  ```bash
  python test.py --model tune_llama
  python test.py --model tune_qwen
  python test.py --model tune_llama70
  ```

- **Other collaborative model combinations**  
  ```bash
  python test.py --model s_s_s
  python test.py --model s_b_s
  # Available combinations: s_s_s / s_s_b / s_b_s / b_s_s / b_s_b / b_b_s / b_b_b
  ```

---

## 📋 Full Training and Prediction Workflow

### OSHA Dataset

1. After placing the dataset, train with Instructor:
   ```bash
   python train.py
   # Enter "y" when prompted to generate prediction results
   ```

2. Generate text embeddings:
   ```bash
   python Text_Embedding.py
   ```

3. Run auxiliary prediction:
   ```bash
   cd LLM/
   python 6_1_deepseek.py  # or qwen.py
   ```

4. Run final decision:
   ```bash
   python 2_1_deepseek.py  # or qwen.py
   ```

### Building Code Dataset

1. Train and optimize BERT/RoBERTa:
   ```bash
   python master.py
   ```

2. Generate initial predictions:
   ```bash
   python test1.py --model bert  # or roberta
   ```

3. Generate text embeddings:
   ```bash
   python Text_Embedding.py
   ```

4. Run auxiliary prediction:
   ```bash
   cd "4、小+大+大/"
   python 6_1_deepseek.py  # or qwen.py / llama.py
   ```

5. Run final decision:
   ```bash
   python 2_1_deepseek.py  # or qwen.py
   ```

---

## 🔍 Explore More Model Combinations and Prompt Strategies

- Customize model paths and parameters in `config.py` or in `6_1/` and `2_1/` scripts.
- This repository only provides a subset of prompt examples used in our experiments. Users can further adjust specific strategies based on actual needs in `initial_messages` or or other parts.

---

## 📁 Using Custom Datasets

1. Format your dataset to match our structure (`train.csv`, `val.csv`, `test.csv`)
2. Place them in the appropriate paths
3. Adjust training parameters and strategies
4. Run the corresponding training and inference scripts as shown above

---

## 📜 License

This repository is for academic and research purposes only.
