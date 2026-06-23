import torch
import torch.cuda
from torch.autograd import Function
import math

# ----------------------------------------------------------------------------------------------------------------------
def _compute_softdtw(D, gamma, bandwidth):
    """
    Pure PyTorch implementation of soft-DTW forward pass using anti-diagonal wavefront.
    Works on both CPU and GPU tensors.
    """
    B, N, M = D.shape
    dev = D.device
    dtype = D.dtype
    R = torch.full((B, N + 2, M + 2), float('inf'), device=dev, dtype=dtype)
    R[:, 0, 0] = 0

    inv_gamma = 1.0 / gamma

    for p in range(1, N + M + 1):
        i_start = max(1, p - M)
        i_end = min(N, p - 1)
        if i_start > i_end:
            continue

        idx_list = []
        for i in range(i_start, i_end + 1):
            j = p - i
            if not (bandwidth > 0 and abs(i - j) > bandwidth):
                idx_list.append((i, j))

        if not idx_list:
            continue

        i_vals = torch.tensor([idx[0] for idx in idx_list], device=dev, dtype=torch.long)
        j_vals = torch.tensor([idx[1] for idx in idx_list], device=dev, dtype=torch.long)

        r0 = -R[:, i_vals - 1, j_vals - 1] * inv_gamma
        r1 = -R[:, i_vals - 1, j_vals] * inv_gamma
        r2 = -R[:, i_vals, j_vals - 1] * inv_gamma
        rmax = torch.max(torch.max(r0, r1), r2)
        rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
        softmin = -gamma * (torch.log(rsum) + rmax)
        R[:, i_vals, j_vals] = D[:, i_vals - 1, j_vals - 1] + softmin

    return R

# ----------------------------------------------------------------------------------------------------------------------
def _compute_softdtw_backward(D_, R, gamma, bandwidth):
    """
    Pure PyTorch implementation of soft-DTW backward pass using anti-diagonal wavefront.
    Works on both CPU and GPU tensors.
    """
    B, N, M = D_.shape
    dev = D_.device
    dtype = D_.dtype

    D = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype)
    D[:, 1:N + 1, 1:M + 1] = D_
    E = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype)
    E[:, -1, -1] = 1
    R[:, :, -1] = float('-inf')
    R[:, -1, :] = float('-inf')
    R[:, -1, -1] = R[:, -2, -2]

    R = torch.where(torch.isinf(R), torch.tensor(float('-inf'), device=dev, dtype=dtype), R)

    inv_gamma = 1.0 / gamma

    for p in range(N + M, 0, -1):
        i_start = max(1, p - M)
        i_end = min(N, p - 1)
        if i_start > i_end:
            continue

        idx_list = []
        for i in range(i_start, i_end + 1):
            j = p - i
            if not (bandwidth > 0 and abs(i - j) > bandwidth):
                idx_list.append((i, j))

        if not idx_list:
            continue

        i_vals = torch.tensor([idx[0] for idx in idx_list], device=dev, dtype=torch.long)
        j_vals = torch.tensor([idx[1] for idx in idx_list], device=dev, dtype=torch.long)

        a = torch.exp((R[:, i_vals + 1, j_vals] - R[:, i_vals, j_vals] - D[:, i_vals + 1, j_vals]) * inv_gamma)
        b = torch.exp((R[:, i_vals, j_vals + 1] - R[:, i_vals, j_vals] - D[:, i_vals, j_vals + 1]) * inv_gamma)
        c = torch.exp((R[:, i_vals + 1, j_vals + 1] - R[:, i_vals, j_vals] - D[:, i_vals + 1, j_vals + 1]) * inv_gamma)
        E[:, i_vals, j_vals] = E[:, i_vals + 1, j_vals] * a + E[:, i_vals, j_vals + 1] * b + E[:, i_vals + 1, j_vals + 1] * c

    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    Pure PyTorch implementation of soft-DTW. Works on both CPU and GPU (CUDA) tensors.
    The anti-diagonal wavefront algorithm ensures correct dependency ordering.
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma_t = torch.tensor([gamma], device=dev, dtype=dtype)
        bandwidth_t = torch.tensor([bandwidth], device=dev, dtype=dtype)

        R = _compute_softdtw(D.detach(), gamma_t.item(), bandwidth_t.item())
        ctx.save_for_backward(D, R.clone(), gamma_t, bandwidth_t)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma_t, bandwidth_t = ctx.saved_tensors

        E = _compute_softdtw_backward(D, R.clone(), gamma_t.item(), bandwidth_t.item())
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        assert bx == by
        assert dx == dy
        return _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    avg_cpu = sum(times_cpu) / len(times_cpu)
    avg_gpu = sum(times_gpu) / len(times_gpu)
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    print("  Speedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    torch.manual_seed(1234)

    profile(128, 17, 15, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    profile(512, 256, 256, 2, tol_backward=1e-3)