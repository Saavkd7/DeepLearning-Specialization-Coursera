# Deep Learning Mathematics Glossary

## 1. The Variables (The Characters)

* **$x$ (Input Feature Vector):** The raw data for a single example (e.g., pixel values of one image).
  
    * *Shape:* $(n_x, 1)$
      
* **$X$ (Input Matrix):** The entire training dataset stacked together. Each **column** is one example.
  
    * *Shape:* $(n_x, m)$ where $m$ is the number of examples.
      
* **$y$ / $Y$ (True Label):** The correct answer (e.g., 1 for Cat, 0 for Non-Cat).
  
* **$W$ (Weights):** The parameters the model learns. It determines the importance of each input feature.
  
    * *Shape:* $(n_{neurons}, n_{inputs})$
      
* **$b$ (Bias):** A learnable parameter that shifts the activation function (like the intercept $c$ in $y=mx+c$).
  
    * *Shape:* $(n_{neurons}, 1)$
      
* **$Z$ (Linear Output):** The weighted sum of inputs plus bias.
  
    * *Formula:* $Z = WX + b$
      
* **$A$ (Activation):** The output of a neuron after applying the non-linear "gate." This is what gets passed to the next layer.
  
    * *Formula:* $A = g(Z)$ (where $g$ is the activation function).
      
* **$\hat{y}$ (Prediction):** The final output of the network (usually the activation of the last layer, e.g., $a^{[2]}$).

## 2. The Scorecards (Loss vs. Cost)

* **$L(\hat{y}, y)$ (Loss Function):** Measures the error for a **single** training example.
  
    * *Formula (Binary Cross-Entropy):*
      
        $$L = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$$
  
    * *Meaning:* "How wrong was I on *this specific image*?".
      
* **$J(W, b)$ (Cost Function):** The average error across the **entire** training set ($m$ examples).
  
    * *Formula:*
      
        $$J = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})$$
  
    * *Meaning:* "How wrong is the model *on average*?" We try to minimize this single number.

## 3. The Processes (The Action)

* **Forward Propagation:** The flow of data from Input $\rightarrow$ Output.
  
    * *Step 1 (Linear):* $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$
      
    * *Step 2 (Activation):* $A^{[l]} = g(Z^{[l]})$
      
    * *Goal:* Calculate the prediction $\hat{y}$ and the Cost $J$.
      
* **Backpropagation:** The flow of error from Output $\rightarrow$ Input.
  
    * *Goal:* Calculate the **Gradients** ($dW, db$). It uses the Chain Rule of Calculus to compute "how much did $W$ contribute to the error?".
      
* **Gradient ($dW, db$):** The **Derivative** (slope) of the Cost Function with respect to a parameter.
  
    * *Meaning:* It tells us the direction to move to increase the error.
     
    * *Formula (Output Layer):* $dZ = A - Y$
      
    * *Formula (Weights):* $dW = \frac{1}{m} X dZ^T$.
      
* **Gradient Descent:** The update step that actually changes the weights.
  
    * *Formula:* $W = W - \alpha \cdot dW$
      
    * *Note:* We subtract because we want to go *down* the slope (minimize error). $\alpha$ is the Learning Rate.

## 4. Activation Functions (The Gates)

* **Sigmoid ($\sigma$):** Squeezes numbers between 0 and 1. Used for **Output Layer** (Binary Probabilities).
  
    * *Formula:* $\frac{1}{1+e^{-z}}$
      
* **Tanh:** Squeezes numbers between -1 and 1. Zero-centered. Used for **Hidden Layers**.
  
    * *Formula:* $\frac{e^z - e^{-z}}{e^z + e^{-z}}$
      
* **ReLU:** Sets negatives to 0, keeps positives unchanged. The standard for **Hidden Layers**.
  
    * *Formula:* $\max(0, z)$
      
* **Derivative ($g'(z)$):** The slope of the activation function, used during Backpropagation to pass the gradient through the layer.
* **ReLU:** Sets negatives to 0, keeps positives unchanged. The standard for **Hidden Layers**.
    * *Formula:* $\max(0, z)$
* **Derivative ($g'(z)$):** The slope of the activation function, used during Backpropagation to pass the gradient through the layer.
