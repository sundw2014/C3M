# Learning Certified Control Using Contraction Metric
Pytorch implementation of the CoRL'20 paper "[Learning Certified Control Using Contraction Metric](https://arxiv.org/abs/2011.12569)", by Dawei Sun, Susmit Jha, and Chuchu Fan.

## Requirements
Dependencies include ```torch```, ```tqdm```, ```numpy```, and ```matplotlib```. You can install them using the following command.
```bash
pip install -r requirements.txt
```

## Usage
The script ```main.py``` can be used for learning the controller. Usage of this script is as follows
```
usage: main.py [-h] [--task TASK] [--no_cuda] [--bs BS]
               [--num_train NUM_TRAIN] [--num_test NUM_TEST]
               [--lr LEARNING_RATE] [--epochs EPOCHS] [--lr_step LR_STEP]
               [--lambda _LAMBDA] [--w_ub W_UB] [--w_lb W_LB] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           Name of the model.
  --no_cuda             Disable cuda.
  --bs BS               Batch size.
  --num_train NUM_TRAIN
                        Number of samples for training.
  --num_test NUM_TEST   Number of samples for testing.
  --lr LEARNING_RATE    Base learning rate.
  --epochs EPOCHS       Number of training epochs.
  --lr_step LR_STEP
  --lambda _LAMBDA      Convergence rate: lambda
  --w_ub W_UB           Upper bound of the eigenvalue of the dual metric.
  --w_lb W_LB           Lower bound of the eigenvalue of the dual metric.
  --log LOG             Path to a directory for storing the log.
```

Taking the 8-dimensional quadrotor model as an example, run the following command to learn a controller.
```
mkdir log_QUADROTOR_8D
python main.py --log log_QUADROTOR_8D --task QUADROTOR_8D
```

Run the following command to evaluate the learned controller and plot the results.
```
python plot.py --pretrained log_QUADROTOR_8D/controller_best.pth.tar --task QUADROTOR_8D --plot_type 3D --plot_dims 0 1 2
python plot.py --pretrained log_QUADROTOR_8D/controller_best.pth.tar --task QUADROTOR_8D --plot_type error
```

If you find this project useful, please cite:
```bibtex
@article{sun2020learning,
  title = {Learning certified control using contraction metric},
  author = {Sun, Dawei and Jha, Susmit and Fan, Chuchu},
  booktitle = {Proceedings of the Conference on Robot Learning},
  year = {2020}
}
```
