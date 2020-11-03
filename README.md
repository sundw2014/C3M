# Learning Certified Control Using Contraction Metric

# Requirements
```bash
pip install -r requirements.txt
```

# Usage
Taking the 8-dimensional quadrotor model as an example, run the following command to learn a controller.
```
mkdir log_QUADROTOR_8D
python main.py --log log_QUADROTOR_8D --task QUADROTOR_8D
```

Run the following command to evaluate the learned controller.
```
python plot.py --pretrained log_QUADROTOR_8D/controller_best.pth.tar --task QUADROTOR_8D --plot_type 3D --plot_dims 0 1 2
python plot.py --pretrained log_QUADROTOR_8D/controller_best.pth.tar --task QUADROTOR_8D --plot_type error
```
