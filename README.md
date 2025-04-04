# AHFL: A Resource-Adaptive Approach for Data-Heterogeneity-Aware Federated Learning

Welcome to the repository for our paper: "**AHFL: A Resource-Adaptive Approach for Data-Heterogeneity-Aware Federated Learning**."

## Introduction
Federated Learning (FL) enables collaborative model training while preserving data privacy. However, resource-constrained edge devices and data heterogeneity pose significant challenges. **AHFL** introduces a novel approach to enhance inference accuracy while optimizing resource utilization. It adapts to the heterogeneity of both data and computing resources, improving the overall performance of Federated Learning systems.

## Key Features
- **Data-Centric Grouping**: Identifies heterogeneity levels among devices to enable efficient adaptation based on the data distribution.
- **Adaptive Model Compression**: Balances resource constraints and learning performance using knowledge distillation techniques, reducing model size and improving inference efficiency.
- **Group Distributed Representation**: Mitigates performance degradation caused by model heterogeneity, improving global model robustness.
- **Efficient Resource Utilization**: Reduces computation by 1.5× and model parameters by 3.8× compared to traditional approaches.
- **Improved Model Accuracy**: Enhances global model inference accuracy by 7.47%, while keeping accuracy loss under 2% between global and local models.
  
## PyTorch Implementation
AHFL is implemented in **PyTorch**, making it compatible with various hardware platforms and easy to extend for specific use cases. PyTorch’s dynamic computation graph allows for flexible model training and experimentation, while leveraging its GPU acceleration capabilities to speed up the learning process.

### Dependencies:
Make sure to install the following Python packages:

```bash
pip install -r requirements.txt
```


## Directory Structure
The repository follows a modular structure:
- **/models**: Contains model architectures (e.g., ResNet, MSDNet).
- **/data**: Handles data loading and preprocessing.
- **/utils**: Utility functions for operations like model evaluation and knowledge distillation.
- **/task**: Core tasks including training and validation.
- **/test**: Validation of core functional modules.
- **/base**: Core federated learning logic.

## Usage


### Execution Commands

To run the federated learning training with AHFL, execute the following command:

```bash
python main.py --epochs 100 --batch_size 32 --lr 0.01 --gpu 0 --data cifar10
```

### Running Tests

After training the model, you can evaluate its performance using test scripts. The following are the common test executions:

1. **Global Model Evaluation**:
   Evaluate the trained global model on the test set to measure its accuracy and performance.

   ```python
   sever.execute_round(test_loader)
   ```

2. **Local Model Evaluation**:
   If you want to evaluate the local models after federated training (e.g., each client), you can implement it like this:

   ```python
   validate(local_model, test_loader)
   ```

3. **Validation During Training**:
   During federated training, we perform local validation on each device:

   ```python
   local_validate(model, val_loader)
   ```

4. **Model Compression and Knowledge Distillation Test**:
   To test the model compression and knowledge distillation, ensure that the appropriate methods are triggered during training. These tests can be part of the training process, where models are distilled to smaller sizes while retaining accuracy.

   ```python
   distillation_loss = KDLoss(temperature=3.0, alpha=0.1)
   ```

### Additional Notes:
- **Logging**: We recommend using the `logging` library for better tracking of training progress and results in real-time, especially on distributed systems.
  
  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  logging.info("Federated Learning training started.")
  ```

- **Hyperparameter Tuning**: You can easily modify hyperparameters such as learning rate, batch size, and the number of epochs by adjusting the arguments passed in the main function.
