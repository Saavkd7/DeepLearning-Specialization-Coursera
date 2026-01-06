# Deep Learning Mathematics Cheat Sheet

## 1. Global Notation & Dimensions
| Symbol | Definition | Shape/Dimension | Note |
| :--- | :--- | :--- | :--- |
| $m$ | Number of training examples | Scalar | Size of dataset |
| $n_x$ | Number of input features | Scalar | e.g., $64 \times 64 \times 3$ for images |
| $n^{[l]}$ | Number of nodes in layer $l$ | Scalar | Architecture hyperparameter |
| $X$ | Input Matrix | $(n_x, m)$ | **Columns** are examples |
| $Y$ | True Label Matrix | $(1, m)$ | 1 = True, 0 = False |
| $W^{[l]}$ | Weight Matrix for layer $l$ | $(n^{[l]}, n^{[l-1]})$ | Maps prev layer to current |
| $b^{[l]}$ | Bias Vector for layer $l$ | $(n^{[l]}, 1)$ | Broadcasted across $m$ columns |

| Symbol | Meaning |
| :--- | :--- |
| **$[l]$** | Layer index |
| **$(i)$** | Training example index |
| **$i$ (subscript)** | Neuron/Unit index in a layer |
| **$X$ (columns)** | Individual training examples |

---

## 2. Forward Propagation (Vectorized)
The flow of data from input to output. $A^{[0]} = X$.

### General Layer $l$
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$
*Where $g^{[l]}$ is the activation function for layer $l$.*

---

## 3. Activation Functions & Derivatives
Used to introduce non-linearity ($g(z)$) and for backpropagation ($g'(z)$).

### A. Sigmoid ($\sigma$)
* **Usage:** Output Layer (Binary Classification).
* **Formula:** $g(z) = \frac{1}{1 + e^{-z}}$
* **Derivative:** $g'(z) = a(1 - a)$

### B. Tanh
* **Usage:** Hidden Layers (Zero-centered).
* **Formula:** $g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
* **Derivative:** $g'(z) = 1 - a^2$

### C. ReLU (Rectified Linear Unit)
* **Usage:** Hidden Layers (Standard).
* **Formula:** $g(z) = \max(0, z)$
* **Derivative:** $$
  g'(z) = \begin{cases} 
  0 & \text{if } z < 0 \\
  1 & \text{if } z > 0 
  \end{cases}
  $$

---

## 4. Loss & Cost Functions
Measuring error.

### Loss Function ($L$)
*Error for a **single** example.*
$$L(\hat{y}, y) = -\left( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right)$$

### Cost Function ($J$)
*Average error across **m** examples.*
$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right]$$


---

## 5. Backpropagation (Vectorized)
Calculating gradients for Layer $l$. Computed from right (output) to left (input).

### Step 1: Error at Layer $l$ ($dZ^{[l]}$)
* **For Output Layer ($L$):**
  $$dZ^{[L]} = A^{[L]} - Y$$
* **For Hidden Layers ($l$):**
  $$dZ^{[l]} = (W^{[l+1]T} dZ^{[l+1]}) * g'^{[l]}(Z^{[l]})$$
  *(Note: $*$ denotes element-wise multiplication)*

### Step 2: Gradients for Parameters ($dW, db$)
$$dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}$$
$$db^{[l]} = \frac{1}{m} \text{np.sum}(dZ^{[l]}, \text{axis}=1, \text{keepdims}=True)$$


---

## 6. Optimization (Gradient Descent)
Updating parameters to minimize Cost $J$.

$$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$

* **$\alpha$ (Alpha):** Learning Rate (Hyperparameter).
* **Direction:** Subtract because gradient points *uphill*, we want to go *downhill*.
