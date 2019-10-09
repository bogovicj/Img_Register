import torch
import torch.nn.functional as F 
import numpy as np 
import os


# @torch.no_grad()
def transform_layer(img, phi, device):
    """
    Transformation layer. Phi is a deformable field, and will be changed to a position field.
    Args:
        img: images in shape [batch, channel, x, y, z], only first channel is the raw image
        phi: deformable field in shape [batch, channel, x, y, z]
        device: training device
    """
    phi = phi.cpu()
    phi = phi.detach().numpy() # phi = phi.numpy()
    phi = np.transpose(phi, (0,2,3,4,1))  # [batch, x, y, z, channel]

    # Add a base grid to deformable field
    base_grid = np.meshgrid(np.linspace(0,phi.shape[1]-1,phi.shape[1]), np.linspace(0,phi.shape[2]-1,phi.shape[2]), np.linspace(0,phi.shape[3]-1,phi.shape[3]))
    base_grid = np.asarray(base_grid)  # [channel, y, x, z]
    base_grid = np.transpose(base_grid, (2,1,3,0))  # [x, y, z, channel]
    base_grid = np.expand_dims(base_grid, axis=0)  # [batch, x, y, z, channel]
    phi += base_grid

    # Scale to [-1,1]
    phi_min = phi.min(axis=(1,2,3), keepdims=True)
    phi_max = phi.max(axis=(1,2,3), keepdims=True)
    phi = (phi-phi_min) * 2 / (phi_max-phi_min) -1

    phi = torch.from_numpy(phi).float()
    phi = phi.to(device)

    # Extract the first channel of img
    img = img.cpu()
    img = img.detach().numpy() # img = img.numpy()  # [batch, channel, x, y, z]
    img_split_channel = np.split(img, img.shape[1], axis=1)  # split channel
    img = img_split_channel[0]  # only the first channel is raw img
    img = torch.from_numpy(img).float()
    img = img.to(device)
    
    # Apply deformable field
    warped = F.grid_sample(img, phi)
    warped.requires_grad = True

    return warped


def transform_layer_position(img, phi):
    """
    Transformation layer. Phi is a position field.
    Args:
        img: images in shape [batch, channel, x, y, z], only first channel is the raw image
        phi: deformable field in shape [batch, channel, x, y, z]
    """
    phi = phi.permute(0,2,3,4,1)  # [batch, x, y, z, channel]

    # Scale to [-1,1]
    phi_min = torch.min(phi, dim=1, keepdim=True)
    phi_min = torch.min(phi_min.values, dim=2, keepdim=True)
    phi_min = torch.min(phi_min.values, dim=3, keepdim=True)
    phi_max = torch.max(phi, dim=1, keepdim=True)
    phi_max = torch.max(phi_max.values, dim=2, keepdim=True)
    phi_max = torch.max(phi_max.values, dim=3, keepdim=True)
    phi = (phi-phi_min.values) * 2 / (phi_max.values-phi_min.values) -1

    # Extract the first channel of img
    img = torch.split(img, 1, dim=1)  # split channel, the first channel is raw img
    
    # Apply deformable field
    warped = F.grid_sample(img[0], phi)

    return warped


def cc_loss(output, target):
    '''
    Pearson correlation loss
    '''
    x = output - torch.mean(output)
    y = target - torch.mean(target)
    loss = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
    return -loss


class ImgRegisterNetwork():
    """
    Image registeration network class that wraps model related functions (e.g., training, evaluation, etc)
    """
    def __init__(self, model, criterion, optimizer, device):
        """
        Args:
            model: a deep neural network model (sent to device already)
            criterion: loss function
            optimizer: training optimizer
            device: training device
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def train_model(self, data):
        """
        Train the model
        Args:
            data: training dataset generated by DataLoader
        Return batch-wise training loss
        """
        self.model.train()
        training_loss = 0

        for batch, [img, tmplt] in enumerate(data):
            img = img.to(self.device)
            tmplt = tmplt.to(self.device)

            # Forward
            phi = self.model(img)
            # Apply transformation layer
            # warped = transform_layer(img, phi, self.device)
            warped = transform_layer_position(img, phi)  
            # Calculate loss
            loss = self.criterion(warped, tmplt)
            training_loss += loss.item()
            # Zero the parameter gradients
            self.optimizer.zero_grad()                    
            # Backward
            loss.backward()
            # Update weights
            self.optimizer.step()

        print("Batch-wise training loss for current epoch is {}".format(training_loss/(batch+1)))
        return training_loss/(batch+1)


    def eval_model(self, data):
        """
        Evaluate the model
        Args:
            data: evaluation dataset generated by DataLoader
        Return batch-wise evaluation loss
        """
        self.model.eval()
        eval_loss = 0

        for batch, [img, tmplt] in enumerate(data):
            with torch.no_grad():  # Disable gradient computation
                img = img.to(self.device)
                tmplt = tmplt.to(self.device)
                phi = self.model(img)
                # warped = transform_layer(img, phi, self.device)
                warped = transform_layer_position(img, phi)  
                loss = self.criterion(warped, tmplt)
                eval_loss += loss.item()

        print("Batch-wise evaluation loss for current epoch is {}".format(eval_loss/(batch+1)))
        return eval_loss/(batch+1)


    def save_model(self, path, epoch, entire=False):
        """
        Save the model to disk
        Args:
            path: directory to save the model
            epoch: epoch that model is saved
            entire: if save the entire model rather than just save the state_dict
        """
        if not os.path.exists(path):
            os.mkdir(path)
        if entire:
            torch.save(self.model, path+"/whole_model_epoch_{}.pt".format(epoch))
        else:
            # torch.save(self.model.state_dict(), path+"/model_ckpt_{}.pt".format(epoch))
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'criterion': self.criterion},
                        path+"/model_ckpt_{}.pt".format(epoch))
    

    def test_model(self, checkpoint, img, tmplt, input_sz):
        """
        Test the model on new data
        Args:
            checkpoint: saved checkpoint
            img: testing data (moving image)
            tmplt: template (fixed image)
            input_sz: network input size in (x,y,z) (network input is [batch, channel, x, y, z])
        """
        assert img.shape == tmplt.shape, "moving image doesn't match fixed template shape!"

        ckpt = torch.load(checkpoint)
        # self.model.load_state_dict(ckpt)
        self.model.load_state_dict(ckpt['model_state_dict'])

        self.model.eval()

        phi = np.zeros((3, img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
        for row in range(0, img.shape[0], input_sz[0]):
            for col in range(0, img.shape[1], input_sz[1]):
                for vol in range(0, img.shape[2], input_sz[2]):
                    # Generate 
                    patch_img = np.zeros((1, 2, input_sz[0], input_sz[1], input_sz[2]), dtype=img.dtype)
                    patch_img[0,0,:,:,:] = img[row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]]
                    patch_img[0,1,:,:,:] = tmplt[row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]]
                    patch_img = torch.from_numpy(patch_img).float()
                    patch_img = patch_img.to(self.device)
                    # Apply model
                    patch_phi = self.model(patch_img)
                    patch_phi = patch_phi.cpu()
                    patch_phi = patch_phi.detach().numpy()
                    phi[:, row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]] = patch_phi[0,:,:,:,:]

        return phi
