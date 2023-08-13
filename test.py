import torch
import numpy as np
from pytransform3d.transformations import (
    transform_log_from_transform,
    transform_from_exponential_coordinates,
    exponential_coordinates_from_transform_log,
)
T = torch.Tensor(
       [[-0.7384057045,  0.3520625532, -0.5751599669, -0.1840534210],
        [ 0.3333132863, -0.5508944392, -0.7651259303, -0.1836946011],
        [-0.5862244964, -0.7566816807,  0.2894364297,  0.9952554703],
        [ 0.0000000000,  0.0000000000,  0.0000000000,  1.0000000000]])

torch.set_printoptions(precision=15)
vee_log_T = exponential_coordinates_from_transform_log(transform_log_from_transform(T))
print(torch.Tensor(vee_log_T))
eq_T = transform_from_exponential_coordinates(vee_log_T)
assert np.isclose(
    T,
    eq_T,
    rtol=1e-5,
    atol=1e-5, 
).all()

#R_rectified, _ = torch.linalg.qr(T[:, :3, :3], mode="reduced")
#T[:, :3, :3] = R_rectified
#eq_T = transform_log_from_transform(transforms.se3_log_map(T))


from pytorch3d import transforms
T = torch.Tensor([[[-0.7384057045,  0.3333132863, -0.5862244964,  0.0000000000],
         [ 0.3520625532, -0.5508944392, -0.7566816807,  0.0000000000],
         [-0.5751599669, -0.7651259303,  0.2894364297,  0.0000000000],
         [-0.1840534210, -0.1836946011,  0.9952554703,  1.0000000000]]])
log_T = transforms.se3_log_map(T)
print(log_T.numpy().tolist())
eq_T = transforms.se3_exp_map(log_T)
