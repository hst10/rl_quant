from time import time
from typing import (Tuple, Optional)
import numpy as np
import torch

"""
Performance below is measured in seconds (total running time)
num_trials, num_runs = 50, 10000

My laptop
  xbar_row, xbar_col, xbar_row_size, xbar_col_size = 1, 3, 2, 2	
    CPU:
      loops:   mean = 0.9268006563186646, std = 0.32272087791433063
      unfold:  mean = 0.10981571197509765, std = 0.02907392144346045
      reshape: mean = 0.07461658477783203, std = 0.013616791471359198

  xbar_row, xbar_col, xbar_row_size, xbar_col_size = 1, 3, 2, 2
    CPU:
      loops:   mean = 0.7654950284957885, std = 0.09917115379335989
      unfold:  mean = 0.1550894021987915, std = 0.01610656256548656
      reshape: mean = 0.1340224027633667, std = 0.0200046938565212

My Workstation
  xbar_row, xbar_col, xbar_row_size, xbar_col_size = 1, 3, 2, 2
    CPU:
      loops:   mean = 0.9029212379455567, std = 0.001952939842758984
      unfold:  mean = 0.28134989261627197, std = 0.00041601104547914685
      reshape: mean = 0.2649233341217041, std = 0.0006611289025128836
    GPU:
      loops:   mean = 1.316331057548523, std = 0.001990797806292587
      unfold:  mean = 0.24577868461608887, std = 0.0007793928691277862
      reshape: mean = 0.23635016441345214, std = 0.0008632943433289636
  xbar_row, xbar_col, xbar_row_size, xbar_col_size = 3, 4, 64, 128
    GPU:
      loops:   mean = 5.406575045585632, std = 0.01800940834636495
      unfold:  mean = 0.24997889041900634, std = 0.0008831850115342929
      reshape: mean = 0.23831791400909424, std = 0.000835532187633969
"""


def init_tensors(xbar_row: int, xbar_col: int, xbar_row_size: int, xbar_col_size: int,
                 device: Optional[torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize tensors."""
    x_bars = torch.zeros(xbar_row, xbar_col, xbar_row_size, xbar_col_size, dtype=torch.int, device=device)
    weight_xbar = torch.randint(low=-100, high=100, size=(xbar_row*xbar_row_size, xbar_col*xbar_col_size),
                                dtype=torch.int, device=device)
    return x_bars, weight_xbar


def run_impl_using_explicit_loops(x_bars: torch.Tensor, weight_xbar: torch.Tensor, num_runs: int = 1) -> float:
    """ Simulate original implementation using explicit loops.
    Example shapes:
        x+bars.shape = torch.Size([1, 3, 64, 64])
        weight_xbar.shape = torch.Size([64, 192])

    This implementation uses '+=' assignment operator instead of '=' in the original implementation.
    Returns:
        Total running time in seconds.
    """
    xbar_row, xbar_col, xbar_row_size, xbar_col_size = x_bars.shape

    start_tm = time()
    for run_id in range(num_runs):
        for i in range(xbar_row):
            for j in range(xbar_col):
                x_bars[i, j] += weight_xbar[
                    i * xbar_row_size: (i + 1) * xbar_row_size,
                    j * xbar_col_size: (j + 1) * xbar_col_size
                ]
    return time() - start_tm


def run_impl_using_unfold(x_bars: torch.Tensor, weight_xbar: torch.Tensor, num_runs: int = 1) -> float:
    """Implementation based on 'unfold' operator.
    Returns:
        Total running time in seconds.
    """
    start_tm = time()
    xbar_row, xbar_col, xbar_row_size, xbar_col_size = x_bars.shape
    for run_id in range(num_runs):
        x_bars += weight_xbar.unfold(0, xbar_row_size, xbar_row_size).unfold(1, xbar_col_size, xbar_col_size)
    return time() - start_tm


def run_impl_using_reshape(x_bars: torch.Tensor, weight_xbar: torch.Tensor, num_runs: int = 1) -> float:
    """Implementation using 'reshape' with strides.
    Understanding this implementation:
        >>> import torch
        >>> t = torch.arange(1, 25).reshape(4, 6)
        >>> t3 = torch.as_strided(t, (2, 3, 2, 2), (12, 2, 6, 1))
    Returns:
        Total running time in seconds.
    I have not done extensive performance testing, but based on one config this is as fast as implementaion based on
    unfolding (for not too small inputs), which is a bit easier to understand.
    """
    start_tm = time()
    xbar_row, xbar_col, xbar_row_size, xbar_col_size = x_bars.shape
    for run_id in range(num_runs):
        x_bars += torch.as_strided(
            weight_xbar,
            (xbar_row, xbar_col, xbar_row_size, xbar_col_size),
            (weight_xbar.shape[1] * xbar_row_size, xbar_col_size, weight_xbar.shape[1], 1)
        )
    return time() - start_tm


def time_implementation(x_bars: torch.Tensor, weight_xbar: torch.Tensor,
                        func_impl: callable,
                        num_trials: int = 1, num_runs: int = 1):
    """Run and measure performance, report mean and standard deviation."""
    # run warm-up iterations
    for trial in range(5):
        func_impl(x_bars, weight_xbar, num_runs)

    # run benchmark iterations
    exec_times = [0] * num_trials
    for trial in range(num_trials):
        exec_times[trial] = func_impl(x_bars, weight_xbar, num_runs)

    #
    print("mean = {}, std = {}".format(np.mean(exec_times), np.std(exec_times)))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # num_trials, num_runs = 50, 10000
    # xbar_row, xbar_col, xbar_row_size, xbar_col_size = 1, 3, 2, 2

    # num_trials, num_runs = 50, 10000
    # xbar_row, xbar_col, xbar_row_size, xbar_col_size = 1, 3, 64, 64

    num_trials, num_runs = 50, 10000
    xbar_row, xbar_col, xbar_row_size, xbar_col_size = 3, 4, 64, 128

    # Create tensors  and verify all tensors are distinct.
    x_bars1, weight_xbar = init_tensors(xbar_row, xbar_col, xbar_row_size, xbar_col_size, device)
    x_bars2, x_bars3, x_bars = x_bars1.clone(), x_bars1.clone(), x_bars1.clone()
    assert torch.equal(x_bars1, x_bars2) and torch.equal(x_bars2, x_bars3)
    assert id(x_bars1) != id(x_bars2) and id(x_bars2) != id(x_bars3) and id(x_bars3) != id(x_bars)

    print(f"x_bars.shape = {x_bars1.shape}")
    print(f"weight_xbar.shape = {weight_xbar.shape}")

    # Check correctness of implementation using one iteration
    run_impl_using_explicit_loops(x_bars1, weight_xbar, 1)
    run_impl_using_unfold(x_bars2, weight_xbar, 1)
    run_impl_using_reshape(x_bars3, weight_xbar, 1)
    assert not torch.equal(x_bars1, x_bars) and not torch.equal(x_bars2, x_bars) and not torch.equal(x_bars3, x_bars)

    if not torch.equal(x_bars1, x_bars2) or not torch.equal(x_bars2, x_bars3):
        print("Tensors do not match, implementations differ.")
        exit(1)

    # Run performance evaluation, check implementation correctness as well.
    x_bars1, weight_xbar = init_tensors(xbar_row, xbar_col, xbar_row_size, xbar_col_size, device)
    x_bars2, x_bars3, x_bars = x_bars1.clone(), x_bars1.clone(), x_bars1.clone()
    time_implementation(x_bars1, weight_xbar, run_impl_using_explicit_loops, num_trials, num_runs)
    time_implementation(x_bars2, weight_xbar, run_impl_using_unfold, num_trials, num_runs)
    time_implementation(x_bars3, weight_xbar, run_impl_using_reshape, num_trials, num_runs)

    if not torch.equal(x_bars1, x_bars2) or not torch.equal(x_bars2, x_bars3):
        print("Tensors do not match, implementations differ.")
        exit(1)


if __name__ == '__main__':
    main()
