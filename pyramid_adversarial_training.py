import torch
import torch.nn.functional as F

H=224
LR=3/255
M=[20,10,1]
S=[32,16,1]
N_STEPS=5

def get_attacked_image(model, loss_fn, image):
  batch_size = image.shape[0]
  
  def get_perturbed_image(deltas):
    return image + sum(
      m*F.upsample(delta, (H,H), mode='nearest')
      for (m, delta) in zip(M, deltas))
  
  def get_perturbed_loss(deltas):
    return loss_fn(model(get_perturbed_image(deltas)))
  
  deltas = [torch.zeros(*(batch_size, 3, H//s, H//s), requires_grad=True).cuda() for s in S]
  for _ in range(N_STEPS):
    step_loss = get_perturbed_loss(deltas)
    deltas_grad = torch.autograd.grad(step_loss, deltas)
    deltas = [delta - LR * torch.sign(deltas_grad[i]) for (i, delta) in enumerate(deltas)]
    
  # TODO(kylesargent): HACK to scale to min max of batch in lieu of known clipping values.
  perturbed_image = get_perturbed_image(deltas)
  
  perturbed_image = torch.maximum(
    perturbed_image,
    image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0])
  
  
  perturbed_image = torch.minimum(
    perturbed_image,
    image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0])
  
  return perturbed_image