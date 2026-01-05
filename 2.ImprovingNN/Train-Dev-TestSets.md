# Applied Deep Learning: Train / Dev / Test Sets

## 1. Executive Summary
Deep Learning is a highly **iterative** process (Idea $\rightarrow$ Code $\rightarrow$ Experiment). It is almost impossible to guess the perfect hyperparameters (layers, learning rate, etc.) on the first try. To speed up this cycle, you must organize your data into three distinct sets: **Training**, **Development (Dev)**, and **Test**.
* **Crucial Shift:** In the era of Big Data, the traditional "70/30" split is obsolete. We now often use splits like **98/1/1**.

## 2. Technical Deep Dive

### The Three Sets
1.  **Training Set:** The data the model learns from (adjusts parameters $W, b$).
2.  **Dev Set (Hold-out Cross Validation):** The data used to **tune hyperparameters**. You train multiple models and use the Dev set to select the "winner."
3.  **Test Set:** The data used *only once* at the very end to get an **unbiased estimate** of performance. You cannot make decisions based on this, or you will overfit to it.

### The Ratio Shift: Small Data vs. Big Data
* **Old Era (Small Data):** If you had 100 to 10,000 examples, the rule of thumb was **60% Train / 20% Dev / 20% Test** (or 70/30).
* **Modern Era (Big Data):** If you have 1,000,000 examples, you don't need 20% (200,000) for the Dev set. You only need enough to statistically determine which algorithm is better (e.g., 10,000 examples).
    * *New Ratio:* **98% Train / 1% Dev / 1% Test** (or even 99.5% / 0.25% / 0.25%).


### The "Distribution" Rule
A common modern problem is **Data Mismatch**.
* *Example:* Your training data is high-quality "Web Crawled" images (Clear, Professional). Your actual app users upload "Mobile" images (Blurry, Amateur).
* **The Golden Rule:** Ensure your **Dev and Test sets come from the same distribution**.
    * *Why?* The Dev set is the "target" you are aiming for. If you aim for one target (Web images) but are tested on another (Mobile images), you will miss.

## 3. "In Plain English"

### The "Olympic Team" Analogy
* **Training:** The daily practice sessions where athletes improve their skills.
* **Dev Set:** The **Qualifying Heats**. You run different strategies here to see which athletes make the cut for the final team.
* **Test Set:** The **Olympic Games**. You run the race once to see how fast you actually are. You can't go back and change your training after the race is run.
* *Big Data Nuance:* If you have 1 million athletes, you don't need 200,000 heats to find the best ones. You just need enough heats to be fair.

## 4. Expert Nuance

### Is the "Test Set" Mandatory?
No.
* If you don't need an unbiased estimate of performance (e.g., you just want the best model for production), you can skip the Test set.
* **Terminology Warning:** Many people use only Train/Dev but call the Dev set the "Test Set." This is technically incorrect (because they are overfitting to it), but it is common practice in industry.
