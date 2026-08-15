# Lecture Machine Learning - Foundations and Algorithms

**Links**
- [Ilias](https://ilias.studium.kit.edu/ilias.php?baseClass=ilrepositorygui&cmdNode=xs:mo&cmdClass=ilobjcoursegui&ref_id=2913524&item_ref_id=0)

## Submission Info

Upload only the edited Jupyter notebook (`.ipynb`) as a raw file to Ilias.
Do not upload any other files or folders, and do not zip the notebook.

To ensure auto-grading works smoothly, please note the following:

- The notebook must have the filename "ex_00_intro.ipynb"
- **Upload only the Jupyter Notebook** on Ilias as a single file (not as a zip!). **Do not include any additional files**.
- Before submitting a notebook, test that everything runs without errors from start to finish.
- **Cells marked with "##### DO NOT CHANGE #####" must not be edited or deleted**
  - There may be seemingly empty cells that are also marked with "##### DO NOT CHANGE #####". These must also not be edited or deleted.
  - If you do modify them, auto-grading will not work and you will receive no points.
  - We will be strict about this and make no exceptions if someone modifies cells clearly marked as read-only!
- Your solution must be entered in the correct cell (marked with "# YOUR CODE HERE").
  - Please **delete the NotImplementedError**
- Generally, **do not delete any cells** and **do not add any cells**. The cells where your solution should be entered already exist (marked with "# YOUR CODE HERE").
- The Jupyter notebooks have inline tests (visible to you) that check your result for basic correctness.
  - These are primarily for you to identify and correct errors.
  - However, the inline tests you can see in the notebook are not the tests used for grading!
  - Passing the inline tests is a necessary but not sufficient condition to receive points!

**If you fail to comply with these rules, your submission may not be graded and you may receive no points for this exercise!**

## Setup

**Requirements**

- Docker / Podman / other container tooling
- Visual Studio Code

Open the project in Visual Studio Code as a DevContainer.
-> When you execute the first cell, VS Code will prompt you to select a Python kernel. Select `Python Environments... > ml-ss26-assignments (.venv/bin/python)`.

**References**

- https://code.visualstudio.com/docs/devcontainers/containers#_getting-started
- https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container

## Exercise Summaries

Six exercises are completed so far, `exercises/task1` through `exercises/task6`. Each folder is a self-contained `uv` project with its own `pyproject.toml`, `uv.lock`, and notebook. The summaries below describe what each notebook implements, not the grading rubric.

### Task 1: Regression and Classification

The notebook covers ridge regression, a generative classifier, and logistic regression. For ridge regression, I derived and implemented the closed-form weight solution and plotted the training and test error against the regularization strength. The generative classifier part implements a Naive Bayes model with maximum-likelihood parameter estimation and looks at what happens when the conditional-independence assumption is violated. The last part implements batch gradient descent for logistic regression from scratch, including the loss and gradient computation.

### Task 2: Multiclass Classification and Kernel Methods

This one has three separate models. First, a multiclass softmax classifier: a numerically stable softmax (subtracting the max before exponentiating), the categorical negative log-likelihood, and the full training loop. Second, kernel ridge regression with a Gaussian kernel: computing the kernel vector and kernel matrix, then the closed-form prediction, evaluated with mean squared error on held-out data. Third, a feature-based SVM with hinge loss, where I implemented both the hinge loss objective and its gradient and compared hinge loss against logistic loss on the same data. The last section is model selection for the kernel ridge regression, choosing the kernel bandwidth by validation performance.

### Task 3: Model Selection, Bayesian Linear Regression, Gaussian Processes

Three parts. Model selection for polynomial regression: hold-out validation and k-fold cross-validation for picking the polynomial degree, plus a look at how partition size affects the selection. Bayesian linear regression: the sequential (online) Bayes update as data arrives, the predictive distribution, and an out-of-distribution robustness comparison against plain ridge regression, since the posterior predictive gives calibrated uncertainty outside the training range while ridge does not. Gaussian processes: implementing the Gaussian (RBF) kernel, deriving the GP predictive distribution, and sampling functions from the GP prior and posterior.

### Task 4: Neural Networks

This exercise builds a neural network from the ground up before touching PyTorch. The first part works through forward and backward passes on a small computation graph by hand, deriving and implementing the gradients with respect to matrix and vector inputs. The second part is a full neural network classifier implemented from scratch in NumPy: weight initialization, forward pass, backpropagation, and a plain gradient-descent optimizer, trained end to end without any autodiff library. The third part switches to PyTorch and trains a convolutional classifier on MNIST, including the conv architecture, the cross-entropy loss on logits, and the training step, then inspects individual predictions.

### Task 5: Transformers

The main part of this exercise implements a transformer from scratch on top of PyTorch's tensor operations: token embeddings, sinusoidal positional encoding, multi-head attention, the feed-forward block, layer normalization, and the encoder and decoder blocks, assembled into a full encoder-decoder transformer with training and inference loops. The second part is theoretical: comparing forward KL and reverse KL divergence on a multi-modal target distribution, looking at how each behaves when fitting data, and relating both to the entropy of the resulting distribution. Trained model checkpoints are saved under `exercises/task5/ex_05_transformers/model/`.

### Task 6: Gaussian Mixture Models and Variational Autoencoders

Two models. Expectation-Maximization for a GMM: the log joint density, the log-likelihood, the E-step (responsibility computation) and M-step (updating the mixture weights, means, and covariances), the full EM loop, sampling from the fitted mixture, and a short discussion of EM's convergence properties (local optima, monotonic likelihood increase). The VAE part implements the reparameterization trick, the ELBO loss (reconstruction term plus KL term), trains the VAE, samples from the learned latent space, and looks at the effect of a fixed versus learned decoder variance on the reconstructions.
