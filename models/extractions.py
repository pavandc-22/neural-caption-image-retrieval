import torch
import torch.nn as nn
from torchvision import models

class VGG19Extractor(nn.Module):
    def __init__(self):
        super(VGG19Extractor, self).__init__()
        # Load the pre-trained VGG-19 model
        vgg19 = models.vgg19(pretrained=True)
        
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        
        # The VGG-19 classifier has 7 layers. 
        # Index 0 is fc6, Index 3 is fc7. Index 6 is the final 1000-class output.
        # We slice the classifier to keep everything up to the fc7 layer and its ReLU activation (index 4).
        self.classifier = nn.Sequential(*list(vgg19.classifier.children())[:5])
        
        # Freeze the weights so we don't accidentally train VGG-19
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):

        is_ten_crop = (x.dim() == 5)
        
        if is_ten_crop:
            batch_size, n_crops, c, h, w = x.size()
            # Flatten the crops into the batch dimension so VGG can process them
            x = x.view(-1, c, h, w)
            
        # Pass through convolutional layers
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Pass through fully connected layers up to fc7
        x = self.classifier(x) # Output shape will be (Batch_Size * 10, 4096)
        
        if is_ten_crop:
            # Reshape back and average the 10 crops into a single 4096-dimensional vector per image
            x = x.view(batch_size, n_crops, -1).mean(dim=1)
            
        return x

# Example usage:
# extractor = VGG19Extractor()
# image_features = extractor(image_tensor_from_dataloader)