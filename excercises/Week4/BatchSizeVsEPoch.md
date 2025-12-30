# Deep Learning Concepts: Batch Size vs. Epoch

## 1. The Definitions
* **Batch Size:** The number of training examples the model "sees" before it updates the weights **once** (one iteration).
* **Epoch:** One complete pass through the **entire** training dataset.

## 2. The Relationship (The Formula)
An epoch is made up of many batches.

$$\text{Iterations per Epoch} = \frac{\text{Total Examples (m)}}{\text{Batch Size}}$$

## 3. Concrete Example
Imagine you have **1,000** images ($m=1000$).

* **Scenario A (Batch Gradient Descent):**
    * **Batch Size:** 1000 (You use all data at once).
    * **Result:** 1 Iteration = 1 Epoch.
    * *Note:* This is what was used in the previous code examples.

* **Scenario B (Mini-Batch Gradient Descent):**
    * **Batch Size:** 100.
    * **Calculation:** $1000 / 100 = 10$.
    * **Result:** It takes **10 Iterations** (10 weight updates) to complete just **1 Epoch**.
