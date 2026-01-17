# My Diary

Today  **Saturday, 17 Jan 2026**.

## General Counter
It has been    **153** days since I started this diary.

# Repository Counter

Day **25** From I started this repository


# Expert Guide: L2 Regularization vs. Dropout

## 1. Executive Summary: The Rule of Thumb
* **L2 Regularization:** This is your **default**. You should almost always include a small amount of L2 regularization (weight decay) in any network to ensure numerical stability and prevent basic overfitting. It is the "standard" approach.
* **Dropout:** This is your **heavy artillery**. Use this primarily for **Computer Vision** or extremely large fully-connected networks where you clearly see high variance (overfitting) that L2 alone cannot fix. It comes with the cost of a jittery loss function.

## 2. Case Studies: When to use which?

### Case A: Structured Data / Standard Problems (Use L2)
* **Scenario:** You are predicting housing prices or customer churn using a standard dataset.
* **Why L2?** L2 keeps the Cost Function $J$ well-defined. You can trust your "Loss vs. Iteration" plots to debug gradient descent.
* **Why Not Dropout?** Dropout introduces noise. If your model isn't overfitting massively, Dropout might just slow down training without adding value.

### Case B: Computer Vision / Massive Parameters (Use Dropout)
* **Scenario:** You are training a Convolutional Neural Network (CNN) to classify images (pixels = high dimensionality).
* **Why Dropout?** In vision, you rarely have "enough" data relative to the input size. Overfitting is the main enemy. Dropout forces the network to learn robust features (e.g., recognizing a cat even if the ear is obscured) rather than memorizing specific pixels.
* **Strategy:** Apply Dropout to the layers with the most parameters (usually the Fully Connected layers at the end), but not to the input layer.

---

## 3. Raw Code Implementation (NumPy)

This snippet demonstrates how to implement both techniques in a modular way.

### Option 1: L2 Regularization (The Default)

**Key Concept:** Add a penalty to the Cost and add a "decay" term to the Gradients.

```python
import numpy as np

def compute_cost_with_l2(A3, Y, parameters, lambd):
    """
    Computes cost with L2 Penalty.
    Formula: Cost + (lambda / 2m) * sum(W^2)
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    # 1. Standard Cross-Entropy Cost
    cross_entropy_cost = -np.sum(np.multiply(Y, np.log(A3)) + np.multiply(1-Y, np.log(1-A3))) / m
    
    # 2. The L2 Penalty
    L2_cost = (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_cost
    return cost

def backward_prop_with_l2(cache, lambd):
    """
    Computes gradients with Weight Decay.
    Formula: dW = (from_backprop) + (lambda / m) * W
    """
    (Z1, A1, W1, b1, ..., dZ3) = cache # Unpack cache
    m = A1.shape[1]
    
    # Standard Gradient
    dW3 = (1./m) * np.dot(dZ3, A2.T)
    
    # ADDING L2 TERM (Weight Decay)
    dW3 = dW3 + (lambd / m) * W3 
    
    return dW3
    
 ```python
 def forward_prop_with_dropout(X, parameters, keep_prob=0.8):
    """
    Implements Inverted Dropout.
    """
    # Retrieve W, b...
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    # --- DROPOUT STEP (Layer 1) ---
    # 1. Initialize Mask (Matrix of random numbers)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    
    # 2. Convert to 0/1 (Kill 20% of neurons if keep_prob=0.8)
    D1 = (D1 < keep_prob).astype(int)
    
    # 3. Apply Mask (Shut down neurons)
    A1 = A1 * D1
    
    # 4. Scale Up (Inverted Dropout) to fix expected value
    A1 = A1 / keep_prob
    # ------------------------------
    
    # Store D1 in cache because we need it for Backprop!
    cache = (Z1, A1, W1, b1, D1) 
    
    return A3, cache

def backward_prop_with_dropout(dA1, cache, keep_prob=0.8):
    """
    Implements Backprop for Dropout.
    You must zero-out the gradient for neurons that were killed in Forward Prop.
    """
    (Z1, A1, W1, b1, D1) = cache
    
    # --- DROPOUT BACKWARD ---
    # 1. Apply the SAME mask from forward prop
    dA1 = dA1 * D1
    
    # 2. Scale Up (Match the forward scaling)
    dA1 = dA1 / keep_prob
    # ------------------------
    
    # Now continue with standard backprop
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1./m) * np.dot(dZ1, X.T)
    
    return dW1
