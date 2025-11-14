import torch
import torch.distributed as dist
from typing import Dict, Optional


class TopKCompressorState():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    def __init__(self, rank=0, epochs=100, final_ratio=0.01, warmup_ratio=0.05):
        self.residuals = {}
        self.current_epoch = 0
        self.ratio = final_ratio
        self.final_ratio = final_ratio
        self.warmup_epoch = epochs * warmup_ratio
        self.name = 'topk'
        self.rank = rank
        # self.residual_decay = 0.99  # Decay residuals slightly each epoch

    def update(self):
        self.current_epoch += 1
        # current_ratio = 1.00 - (self.current_epoch / self.warmup_epoch) * (1.00 - self.final_ratio)
        # self.ratio = max(current_ratio, self.final_ratio)

        
    def compress(self, tensor, bucket_index):
        with torch.no_grad():
            if bucket_index not in self.residuals:
                self.residuals[bucket_index] = torch.zeros_like(tensor)
            elif self.residuals[bucket_index].shape != tensor.shape:
                # 如果形状不匹配，重新初始化残差
                self.residuals[bucket_index] = torch.zeros_like(tensor)

            
            # Add residual to current gradient
            grad = tensor + self.residuals[bucket_index]
            
            # Top-k compression
            numel = tensor.numel()
            k = max(int(numel * self.ratio), 1)
            
            # Find top-k elements
            abs_grad = torch.abs(grad)
            values, local_indices = torch.topk(abs_grad, k=k, sorted=False)
            # values, local_indices = f_topk(abs_grad, k)
            # local_indices = local_indices.to(device=tensor.device, dtype=torch.long)
            values = grad[local_indices]
            
            # Update residual (remove selected elements)
            grad_residual = grad.clone()
            # grad_residual[local_indices] = 0
            grad_residual.scatter_(0, local_indices, torch.zeros_like(local_indices, dtype=grad_residual.dtype))
            self.residuals[bucket_index] = grad_residual  
            
            return values, local_indices
        
        
def _topk_hook(state: TopKCompressorState,
               bucket: dist.GradBucket
               ) -> torch.futures.Future[torch.Tensor]:

    if bucket.buffer().numel() == 0:
        # Return a completed future with zero tensor
        fut = torch.futures.Future()
        fut.set_result(torch.zeros_like(buffer))
        return fut 
        
    # Initialize key variables
    group_to_use = dist.group.WORLD
    device = bucket.buffer().device
    world_size = dist.get_world_size(group_to_use)
    
    values, indices = state.compress(tensor = bucket.buffer(),bucket_index = bucket.index())

    
    # Concatenate all values and indices

    if values is not None:
        indices_as_type = indices.to(values.dtype)
        concatenated = torch.cat([indices_as_type, values])
    else:
        values = torch.empty(0, dtype=torch.long, device=device)
        indices = torch.empty(0, dtype=bucket.buffer().dtype, device=device)


    local_size = concatenated.numel()
    max_size = local_size
    
    gathered_tensor = torch.empty(world_size * max_size, dtype=values.dtype, device=device)
    

    def decompress(fut):
            
        # 假设 gathered_tensor 是 [world_size * local_size] 的 1D tensor
        view_tensor = gathered_tensor.view(world_size, local_size)  # [W, L]
        half = local_size // 2

        # 一次性切分索引和值
        indices = view_tensor[:, :half]      # [W, half]
        values  = view_tensor[:, half:]      # [W, half]

        # flatten 并转换类型（一次性）
        merged_indices = indices.reshape(-1).to(torch.long)  # [W * half]
        merged_values  = values.reshape(-1)                  # [W * half]

        # scatter_add_
        buffer = bucket.buffer()
        buffer.zero_()
        buffer.scatter_add_(0, merged_indices, merged_values)
        buffer.div_(world_size)

        return buffer
    # 返回异步 future
    return (
        dist.all_gather_into_tensor(gathered_tensor, concatenated, group=group_to_use, async_op=True)
        .get_future()
        .then(decompress)
    )        



def _allreduce_fut(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = dist.group.WORLD
    tensor = bucket.buffer()
    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def topk_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook(process_group, bucket)
    """
    # return _allreduce_fut(process_group, bucket)
    if process_group.current_epoch < process_group.warmup_epoch:
        return _allreduce_fut(process_group, bucket)
    else:
        return _topk_hook(process_group, bucket)
    