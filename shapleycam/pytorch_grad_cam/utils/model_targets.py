import torch


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class ClassifierOutputSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]

class ClassifierOutputReST:
    def __init__(self, category, temperature=0.9, epsilon=0):
        self.category = category
    def __call__(self, model_output): 
        """
        Computes the loss for the given model output.

        Args:
            model_output (torch.Tensor): The model output logits.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Check the dimensionality of the model output
        if len(model_output.shape) == 1:
            # Convert the target category to a tensor
            target = torch.tensor([self.category], device=model_output.device)
            model_output = model_output.unsqueeze(0)
            return model_output[0][self.category] - torch.nn.functional.cross_entropy(model_output, target)
        else:
            # For batch-wise model output
            target = torch.tensor([self.category] * model_output.shape[0], device=model_output.device)
            return model_output[:,self.category]- torch.nn.functional.cross_entropy(model_output, target)
