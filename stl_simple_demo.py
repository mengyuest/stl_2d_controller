import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from stl_d_lib import *

def build_relu_nn(input_dim, output_dim, hiddens, activation_fn, last_fn=None):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(activation_fn())
    if last_fn is not None:
        layers[-1] = last_fn()
    else:
        del layers[-1]
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.input_dim = 4 # (x, y, th, v)
        self.output_dim = args.nt * 2  # 
        self.model = build_relu_nn(self.input_dim, self.output_dim, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):
        controls = self.model(x).reshape(x.shape[0], self.args.nt, 2)
        controls_0 = torch.tanh(controls[..., 0]) * self.args.w_max
        controls_1 = torch.tanh(controls[..., 1]) * self.args.a_max
        controls_new = torch.stack([controls_0, controls_1], dim=-1)
        return controls_new

def to_np(x):
    return x.detach().cpu().numpy()

def uniform(a, b, size):
    return torch.rand(*size) * (b - a) + a

def generate_trajectories(s, us, nt, dt):
    trajs = [s]
    for ti in range(nt):
        s = trajs[-1]
        new_s = dynamics(s, us[:, ti], dt)
        trajs.append(new_s)
    return torch.stack(trajs, dim=1)

def dynamics(s, u, dt):
    new_x = s[..., 0] + s[..., 3] * torch.cos(s[..., 2]) * dt
    new_y = s[..., 1] + s[..., 3] * torch.sin(s[..., 2]) * dt
    new_th = s[..., 2] + u[..., 0] * dt
    new_v = s[..., 3] + u[..., 1]* dt
    return torch.stack([new_x, new_y, new_th, new_v], dim=-1)

def main():
    VIZ_DIR = "./viz"
    c=input("This code will create a \"%s\" directory here. Do you want to proceed? (Y/n)"%VIZ_DIR)
    if c.upper()!="Y":
        return
    os.makedirs(VIZ_DIR, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    loc_0 = [-1, 9, -1, 9]
    loc_a = [1, 3, 5, 7]
    loc_b = [5, 7, 1, 3]
    loc_c = [6, 8, 6, 8]
    loc_d = [4, 5, 4, 5]
    in_map = Always(0, args.nt, 
        And(
            AP(lambda x: (loc_0[1]- x[..., 0])*(x[..., 0]-loc_0[0])),
            AP(lambda x: (loc_0[3]- x[..., 1])*(x[..., 1]-loc_0[2])),
        )
    )
    reach_a = Eventually(0, args.nt, 
        And(
            AP(lambda x: (loc_a[1]- x[..., 0])*(x[..., 0]-loc_a[0])),
            AP(lambda x: (loc_a[3]- x[..., 1])*(x[..., 1]-loc_a[2])),
        )
    )
    reach_b = Eventually(0, args.nt, 
        And(
            AP(lambda x: (loc_b[1]- x[..., 0])*(x[..., 0]-loc_b[0])),
            AP(lambda x: (loc_b[3]- x[..., 1])*(x[..., 1]-loc_b[2])),
        )
    )
    land_c = Eventually(0, args.nt, 
        Always(0,args.nt,And(
            AP(lambda x: (loc_c[1]- x[..., 0])*(x[..., 0]-loc_c[0])),
            AP(lambda x: (loc_c[3]- x[..., 1])*(x[..., 1]-loc_c[2])),
        ))
    )
    avoid_d = Always(0, args.nt, 
        Not(And(
            AP(lambda x: (loc_d[1]- x[..., 0])*(x[..., 0]-loc_d[0])),
            AP(lambda x: (loc_d[3]- x[..., 1])*(x[..., 1]-loc_d[2])),
        ))
    )

    stl = ListAnd([in_map, Or(reach_a, reach_b), avoid_d, land_c])
    
    n = args.num_samples
    rand_x = uniform(0, 2, (n, 1))
    rand_y = uniform(0, 2, (n, 1))
    rand_th = uniform(0, np.pi/2, (n, 1))
    rand_v = uniform(1, 4, (n, 1))

    x_init = torch.cat([rand_x, rand_y, rand_th, rand_v], dim=-1).cuda()
    
    net = Net(args).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    for iter_i in range(args.niters):
        input_x = x_init.detach()
        controls = net(input_x)
        trajectories = generate_trajectories(input_x, controls, args.nt, args.dt)
        scores = stl(trajectories,args.smoothing_factor)[:, 0]
        acc = torch.mean((scores>0).float())
        loss = torch.mean(torch.relu(0.5 - scores))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_i % args.print_freq == 0:
            print("[%04d|%04d] loss:%.3f accuracy:%.3f"%(
                iter_i, args.niters, loss.item(), acc.item()
            ))
        trajs_np = to_np(trajectories)
        scores_np = to_np(scores)
        if iter_i % args.viz_freq == 0 or iter_i==args.niters-1:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
            for ii, (loc, color) in enumerate([[loc_a, "cyan"], [loc_b, "orange"], [loc_c, "blue"], [loc_d, "gray"]]):
                patch = Polygon([[loc[0], loc[2]], [loc[0], loc[3]], [loc[1], loc[3]], [loc[1], loc[2]]], color=color)
                ax.add_patch(patch)
            for i in range(min(args.num_samples,100)):
                plt.plot(trajs_np[i,:,0], trajs_np[i,:,1], color="green" if scores_np[i]>0 else "red", alpha=0.1)
            plt.xlim(-2, 9)
            plt.ylim(-2, 9)
            plt.axis("scaled")
            plt.savefig("%s/traj_iter%05d.png"%(VIZ_DIR, iter_i) , bbox_inches='tight', pad_inches=0.3)
            plt.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--niters", type=int, default=500)
    add("--print_freq", type=int, default=20)
    add("--save_freq", type=int, default=100)
    add("--viz_freq", type=int, default=100)
    add("--lr", type=float, default=3e-4)
    add("--nt", type=int, default=15)
    add("--dt", type=float, default=0.2)
    add("--net_pretrained_path", "-P", type=str, default=None)
    add("--hiddens", type=int, nargs="+", default=[256, 256])
    add("--num_samples", type=int, default=1000)
    add("--w_max", type=float, default=4.0)
    add("--a_max", type=float, default=4.0)
    add("--smoothing_factor", type=float, default=100.0)
    args = parser.parse_args()
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.3f seconds"%(t2-t1))   