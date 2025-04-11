# AHFL: A resource-adaptive approach for data-heterogeneity-aware federated learning

Welcome to the repository for our paper: "**AHFL: A resource-adaptive approach for data-heterogeneity-aware federated learning**."



## 1. Introduction
Federated Learning (FL) is a revolutionary approach that enables collaborative model training across multiple edge devices while preserving data privacy. However, it faces significant challenges due to resource-constrained edge devices and data heterogeneity.

**AHFL** presents a novel solution to enhance inference accuracy and optimize resource utilization. It is designed to adapt to the heterogeneity of both data and computing resources, thereby improving the overall performance of Federated Learning systems.


### 2.1 Dependencies
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

### Additional Notes:
- **Logging**: We recommend using the `logging` library for better tracking of training progress and results in real-time, especially on distributed systems.


- **Hyperparameter Tuning**: You can easily modify hyperparameters such as learning rate, batch size, and the number of epochs by adjusting the arguments passed in the main function.

