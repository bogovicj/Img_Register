import torch
import torch.nn.functional as F

import numpy as np
import nrrd


def load_dfield( nrrd_file ):
    """
    Load a displacement field from a nrrd file as a numpy array,
    and permutes axes in a way that python likes.

    Specifically, expect an input of shape (3, Nx, Ny, Nz)
    where dfield[0,x,y,z] gives the displacement in physical units
    along the 'x' axis at position (x,y,z)

    Returns an array of shape (Nz, Ny, Nx, 3),
    where out[z,y,x,0] gives the displacement in physical units
    along the 'z' axis at position (z,y,x)

    :param nrrd_file:
    :return:
    """
    dfield, hdr = nrrd.read(nrrd_file)
    #tmp = np.transpose(dfield, [3, 2, 1, 0])
    #dfield_perm = np.flip(tmp, 3)

    #dfield_perm = np.flip( np.transpose(dfield, [3, 2, 1, 0]), 3)
    #spacing = np.diagonal(hdr['space directions'][1:, :])
    #spacing_perm = np.flipud(spacing).copy()

    dfield_perm = np.transpose(dfield, [1, 2, 3, 0])
    spacing = np.diagonal(hdr['space directions'][1:, :])

    return dfield_perm, spacing


def gen_identity_grid(sz):
    mgrid = torch.meshgrid(
            torch.linspace(-1, 1, sz[2]),
            torch.linspace(-1, 1, sz[1]),
            torch.linspace(-1, 1, sz[0]))

    id_grid_stack = torch.stack(mgrid).permute(3, 2, 1, 0).unsqueeze(0)
    return id_grid_stack

def torch_position_to_dfield( grid, spacing=[1.,1.,1.]):
    """
    Input grid is what is usable by torch.nn.functional.grid_sample
    i.e. in pixel coordinates such that

    input grid is shape [1 nz ny nx 3]
    grid[:,:,:,:,0] holds x grid positions
    grid[:,:,:,:,1] holds y grid positions
    grid[:,:,:,:,2] holds z grid positions

    spacing is size [3] holding [z y x] spacings

    :param grid:
    :param spacing:
    :return:
    """

    # grid is [ 1, z, y, x, 3] (dx, dy, dz)
    sz = torch.tensor(grid.squeeze().size())[:3]
    id_grid = gen_identity_grid(sz.numpy())

    displacement_raw = (grid - id_grid).squeeze()

    # make displacement  [z, y, x, 3] (dz, dy, dz)
    displacement = torch.flip( displacement_raw, [3] )

    sz_mul = 2.0 / (sz.float() - 1.0)
    spacing_t = torch.as_tensor(spacing.copy())  # spacing may be a numpy or python array, or torch
    mul = (spacing_t / sz_mul).reshape(1, 1, 1, 3)

    displacement *= mul
    return displacement


def dfield_to_torch_position(dfield, spacing=[1., 1., 1.]):
    """
    Input displacement must be in physical coordinates
    specified by the given spacing parameter.

    input dfield is shape [nz ny nx 3]
    dfield[:,:,:,0] holds z displacements
    dfield[:,:,:,1] holds y displacements
    dfield[:,:,:,2] holds x displacements

    spacing is size [3] holding [z y x] spacings

    :param dfield:
    :param spacing:
    :return:
    """

    # the last dimension must hold the displacement vector

    np.flip( dfield)

    # copies may be necessary to avoid negative stride
    dfield_copy = dfield.astype('=f4')

    # IMPORTANT: The flipping below is because
    # the torch call:
    #   F.grid_sample( im, grid )
    #
    # for an image of size [1, 1, z, y, x]
    # grid must be of size [1, z, y, x, 3]
    # where
    #   grid[:,:,:,:0] holds x displacements
    #   grid[:,:,:,:1] holds y displacements
    #   grid[:,:,:,:2] holds z displacements

    dfield_t = torch.flip(torch.from_numpy(dfield_copy), [3])

    spacing_t = torch.as_tensor(spacing.copy())  # spacing may be a numpy or python array
    spacing_t = torch.flipud(spacing_t)

    mtmp = 2.0 / (torch.as_tensor( dfield_t.size()[:3]).float() - 1.0)

    # mtmp is in z y x order, but it needs to be x y z so:
    mtmp = torch.flipud(mtmp)

    mul = mtmp / spacing_t

    # displacements to pixel coordinates
    dfield_t *= mul

    # the grid torch needs for grid_sample
    identity_grid = torch.meshgrid( torch.linspace(-1, 1, dfield_t.size(2)),
                                    torch.linspace(-1, 1, dfield_t.size(1)),
                                    torch.linspace(-1, 1, dfield_t.size(0)))

    id_grid_stack = torch.stack( identity_grid ).permute(3, 2, 1, 0)
    position_grid = (id_grid_stack + dfield_t)

    return position_grid.unsqueeze(0)


if __name__ == '__main__':
    print('test')

    dfield_f = '/home/john/tmp/t1-head_landmarks_dfield.nrrd'
    im_f = '/home/john/tmp/t1-head.nrrd'
    im_xfm_f = '/home/john/tmp/t1-head_torchXfm.nrrd'

    #dfield, dfield_hdr = nrrd.read(dfield_f)
    dfield, spacing = load_dfield(dfield_f)
    print(spacing)
    print(dfield.shape)

    img, img_hdr = nrrd.read(im_f)

    img_t = torch.from_numpy(img.astype('=f4'))
    img_rs = img_t.unsqueeze(0).unsqueeze(0)

    torch_grid = dfield_to_torch_position(dfield, spacing=spacing)
    print( torch_grid.size() )

    img_warped = F.grid_sample( img_rs, torch_grid, align_corners=True)
    print( img_warped.size() )
    print( img_warped.squeeze().size())

    nrrd.write( im_xfm_f,
                img_warped.squeeze().cpu().numpy().astype('>f4'),
                img_hdr)


