"""© Stefan Hofmeister
"""

import numpy as np
import glob
import os
import pdb
import sys
try:
    import cupy
    import cupyx.scipy.fft as cufft
    import cupyx.scipy.fftpack as cufftpack
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

"""
Refactored by Claude for batch masked deconvolution of 4096x4096 AIA images.
"""
def deconvolve_bid_min_mask(aia_data, psf, masks, iterations=25, tolerance=0.1, bg_batch_size=4, bid_batch_size=40, dtypeuse=np.float32):
    """
    Deconvolve a batch of 4096x4096 AIA images in masked subregions using the BID algorithm.

    Assumes use_gpu=True, large_psf=True, pad=True, estimate_background=True, constrain_positive=True.
    The PSF is 8192x8192 (large_psf=True, i.e., double the image size, allowing scattering over
    the full image range).

    Two FFTs are precomputed once:
      (1) The full-image PSF FFT (8192x8192) — reused for every background estimation.
          The original code zeros the PSF center pixel before the background convolution to remove
          self-contribution. But because the image is also zeroed inside the mask region, and we
          only use the background *inside* the mask region, the self-contribution term
          (psf_center * img_zeroed_in_mask) is zero everywhere we care about. So we can reuse
          the full PSF FFT as-is.
      (2) A truncated PSF FFT sized to the largest mask — reused for every masked deconvolution.

    The masked deconvolution is batched: all subregions are padded into a shared working size
    (2*max_sub_h x 2*max_sub_w), stacked into a 3D array, and the BID FFTs are run as a batch.
    Convergence is tracked per-image; the loop continues until all images have converged or
    the iteration limit is reached.

    Parameters
    ----------
    aia_data : list of numpy 2d arrays
        List of 4096x4096 images.
    psf : numpy 2d array
        The 8192x8192 point spread function.
    masks : list of 1d arrays or lists
        One mask per image. Each mask is [left, bottom, right, top] defining an even-sized subregion.
        The subregion dimensions must be smaller than 2048 in each axis.
    iterations : int
        Maximum number of iterations.
    tolerance : float
        Convergence threshold: stop when max absolute pixel change < tolerance.
    bg_batch_size : int
        Number of images to process simultaneously during background estimation.
        Each image requires ~1.5GB of GPU memory for the 8192x8192 FFT, so set this
        based on available GPU memory. Default 4 (~6GB) is conservative for a 24GB GPU.
    bid_batch_size : int
        Number of images to deconvolve simultaneously in the BID iteration loop.
        Memory scales as ~5 arrays of (batch, work_h, work_w) in float64. Default 8.

    Returns
    -------
    list of numpy 2d arrays
        The deconvolved subregion for each image (cropped to the mask size).
    """
    k = 1.0
    psf = np.copy(psf)
    n_imgs = len(aia_data)

    # -------------------------------------------------------------------------
    # Parse all masks into boundary boxes and find the max subregion size
    # -------------------------------------------------------------------------
    bd_boxes = [np.array(mask) for mask in masks]

    sub_heights = [bd[1] - bd[0] + 1 for bd in bd_boxes]
    sub_widths  = [bd[3] - bd[2] + 1 for bd in bd_boxes]
    max_sub_h = max(sub_heights)
    max_sub_w = max(sub_widths)

    # -------------------------------------------------------------------------
    # (1) Precompute the full-image PSF FFT for background estimation.
    #
    #     With large_psf=True and pad=True, the 4096x4096 image is padded by 2048 on each
    #     side (0.5 * 4096), giving 8192x8192 — matching the 8192x8192 PSF.
    #     The PSF is not padded further in the large_psf branch.
    #     Working size: 8192x8192.
    # -------------------------------------------------------------------------
    psf_bg = np.copy(psf)
    psf_bg = np.roll(np.roll(psf_bg, 8192 // 2, axis=0), 8192 // 2, axis=1)
    psf_bg = cupy.array(psf_bg)
    psf_bg_fft = cupy.fft.rfft2(psf_bg)
    del psf_bg

    # -------------------------------------------------------------------------
    # (2) Precompute the truncated PSF FFT for the masked deconvolution.
    #
    #     After subregion extraction, the original code sets large_psf=True. The truncated
    #     PSF spans 2*sub_size around the PSF center (4096,4096), so scattering over the
    #     full subimage range is included. With large_psf=True, the subimage is padded by
    #     0.5*sub_size on each side, making the working array 2*sub_size x 2*sub_size —
    #     the same size as the truncated PSF.
    #
    #     We use the max subregion size so the FFT is done once and reused.
    # -------------------------------------------------------------------------
    work_h = 2 * max_sub_h
    work_w = 2 * max_sub_w

    psf_trunc = psf[4096 - max_sub_h : 4096 + max_sub_h,
                     4096 - max_sub_w : 4096 + max_sub_w]
    psf_trunc = np.roll(np.roll(psf_trunc, work_h // 2, axis=0), work_w // 2, axis=1)
    psf_trunc = cupy.array(psf_trunc)
    psf_trunc_fft = cupy.fft.rfft2(psf_trunc)
    del psf_trunc

    tolerance_gpu = cupy.array([tolerance])

    # -------------------------------------------------------------------------
    # Background estimation and subregion extraction (minibatched).
    #
    # The 8192x8192 FFTs are batched in groups of bg_batch_size. Each image
    # in the minibatch is padded to 8192x8192 on the CPU, stacked into a 3D
    # array, transferred to GPU, and the forward FFT + multiply + inverse FFT
    # are done as a single batched cuFFT call.
    #
    # Results are stored on CPU as padded working arrays to avoid holding the
    # full (n_imgs, work_h, work_w) on GPU simultaneously.
    # -------------------------------------------------------------------------
    # Store results on CPU: padded working arrays and pad masks
    img_work_cpu = np.zeros((n_imgs, work_h, work_w), dtype=dtypeuse)
    pad_mask_cpu = np.ones((n_imgs, work_h, work_w), dtype=np.bool_)

    # Store per-image padding offsets for extraction at the end
    pad_offsets = []

    for batch_start in range(0, n_imgs, bg_batch_size):
        batch_end = min(batch_start + bg_batch_size, n_imgs)
        batch_n = batch_end - batch_start
        print("bg batch %d-%d" % (batch_start, batch_end - 1))

        # Prepare the minibatch on CPU: zero the mask region, pad to 8192x8192
        bg_input_cpu = np.zeros((batch_n, 8192, 8192), dtype=dtypeuse)
        orig_imgs_cpu = []
        for i, imgi in enumerate(range(batch_start, batch_end)):
            img = np.copy(aia_data[imgi])
            orig_imgs_cpu.append(img)
            bd_box = bd_boxes[imgi]
            img_for_bg = np.copy(img)
            img_for_bg[bd_box[0] : bd_box[1] + 1, bd_box[2] : bd_box[3] + 1] = 0.
            bg_input_cpu[i, 2048:6144, 2048:6144] = img_for_bg

        # Transfer to GPU and run batched FFT
        bg_input_gpu = cupy.array(bg_input_cpu)
        del bg_input_cpu
        bg_fft = cupy.fft.rfft2(bg_input_gpu, axes=(-2, -1))
        del bg_input_gpu
        bg_fft *= psf_bg_fft[None, :, :]
        bg_result = cupy.fft.irfft2(bg_fft, axes=(-2, -1))
        del bg_fft

        # Unpad, subtract background, extract subregions, store on CPU
        for i, imgi in enumerate(range(batch_start, batch_end)):
            bd_box = bd_boxes[imgi]
            sub_h = sub_heights[imgi]
            sub_w = sub_widths[imgi]

            img_background = bg_result[i, 2048:6144, 2048:6144]
            img_gpu = cupy.array(orig_imgs_cpu[i]) - img_background

            img_sub = img_gpu[bd_box[0] : bd_box[1] + 1, bd_box[2] : bd_box[3] + 1]
            del img_gpu

            # Pad subimage into the CPU working array (centered)
            pad_top = (work_h - sub_h) // 2
            pad_left = (work_w - sub_w) // 2
            pad_offsets.append((pad_top, pad_left, sub_h, sub_w))

            img_work_cpu[imgi, pad_top : pad_top + sub_h, pad_left : pad_left + sub_w] = cupy.asnumpy(img_sub)
            pad_mask_cpu[imgi, pad_top : pad_top + sub_h, pad_left : pad_left + sub_w] = False
            del img_sub

        del bg_result, orig_imgs_cpu

    # Free the full-image PSF FFT — no longer needed
    del psf_bg_fft

    # -------------------------------------------------------------------------
    # Minibatched BID iterative deconvolution.
    #
    # Process bid_batch_size images at a time through the BID loop.
    # Each minibatch is built on the GPU from CPU storage, run to convergence,
    # then results are extracted back to CPU.
    #
    # cupy.fft.rfft2 with axes=(-2, -1) runs the minibatch as a single
    # batched cuFFT call.
    #
    # Memory: ~5 arrays of (bid_batch_size, work_h, work_w) in float64.
    # -------------------------------------------------------------------------
    img_decons = []

    for batch_start in range(0, n_imgs, bid_batch_size):
        batch_end = min(batch_start + bid_batch_size, n_imgs)
        batch_n = batch_end - batch_start
        print("BID batch %d-%d" % (batch_start, batch_end - 1))

        # Transfer this minibatch to GPU
        img_work_mb = cupy.array(img_work_cpu[batch_start:batch_end])
        pad_mask_mb = cupy.array(pad_mask_cpu[batch_start:batch_end])

        img_decon_mb = img_work_mb.copy()
        active = cupy.ones(batch_n, dtype=cupy.bool_)

        for n_iter in range(iterations):
            img_decon_last = img_decon_mb.copy()

            # Batched FFT: convolve all images in minibatch with the truncated PSF
            img_decon_con = cupy.fft.rfft2(img_decon_mb, axes=(-2, -1))
            img_decon_con = img_decon_con * psf_trunc_fft[None, :, :]
            img_decon_con = cupy.fft.irfft2(img_decon_con, axes=(-2, -1))
            img_decon_con[pad_mask_mb] = 0.

            # Derive deviation and adjust
            img_decon_mb -= k * (img_decon_con - img_work_mb)
            img_decon_mb[img_decon_mb < 0] = 0

            # Check per-image convergence
            diff = cupy.abs(img_decon_mb - img_decon_last)
            dev_per_img = diff.reshape(batch_n, -1).max(axis=1)

            newly_converged = active & (dev_per_img <= tolerance_gpu[0])
            if cupy.any(newly_converged):
                for idx in cupy.where(newly_converged)[0]:
                    print("Image %d converged at iteration %d" % (int(idx) + batch_start, n_iter))
                active &= ~newly_converged

            if not cupy.any(active):
                break

            # For converged images, freeze them: restore to their last state so
            # they don't drift from the positivity clamp on subsequent iterations
            for idx in cupy.where(~active)[0]:
                img_decon_mb[int(idx)] = img_decon_last[int(idx)]

        # Extract each subregion from this minibatch and return to CPU
        for i, imgi in enumerate(range(batch_start, batch_end)):
            pad_top, pad_left, sub_h, sub_w = pad_offsets[imgi]
            img_decon = img_decon_mb[i, pad_top : pad_top + sub_h, pad_left : pad_left + sub_w]
            img_decon = cupy.asnumpy(img_decon)
            img_decons.append(img_decon.astype(aia_data[imgi].dtype))

        del img_decon_mb, img_decon_last, img_work_mb, pad_mask_mb

    return img_decons


def deconvolve_bid_min(imgs, psf, iterations = 25, tolerance = .1):
    """
    Deconvolve an image with the point spread function

    Perform image deconvolution on an image with the instrument
    point spread function using the bid algorithm published in 
    Hofmeister et al. (2024), The Basic Iterative Deconvolution: A Fast Instrumental Point-Spread Function Deconvolution Method That Corrects for Light That Is Scattered Out of the Field of View of a Detector, Solar Physics, Volume 299, Issue 6, article id.77
    https://ui.adsabs.harvard.edu/abs/2023arXiv231211784H/abstract

    Parameters
    ----------
    img : 'numpy 2d array'
        An image.
    psf : `~numpy.ndarray
        The point spread function. 
    iterations: `int`
        Maximum number of iterations
    tolerance: 'float'
        The image deconvolution stops when the maximum change from all pixels between the simulated observed image and the observed image is less than TOLERANCE counts.

    Returns
    -------
    `~sunpy.map.Map`
        Deconvolved image

    """
    #At the moment, the mask option only works if the shape of the selected region is smaller than 0.5 * the shape of the image
        
    #this factor determines the speed of convergence, and should be set between [0.1, 1.0]
    k = 1.0
    
    #createa a copy of the image and psf
    psf = np.copy(psf)


    psf = cupy.array(psf)    
    #derive the fourier transform of the psf
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0), psf.shape[1]//2, axis=1)
    psf = np.fft.rfft2(psf) 

    # if we have a gpu, put it to the gpu
    tolerance = cupy.array([tolerance]) 

    pad_mask = None
    img_decons = []
    for imgi, img in enumerate(imgs):
        
        im_size =  np.array(img.shape)
        img = cupy.array(img)

        img = np.pad(img, ((2048,2048), (2048,2048)), constant_values = np.nan)
        img[img == 0] = 0. #np.finfo(img.dtype).tiny

        if pad_mask is None:
            pad_mask = np.isnan(img)  #the pad mask is required in the next loop - at each iteration, the padding has to be restored.
            pad_mask = cupy.array(pad_mask)

        img[pad_mask] = 0.
        psf[np.isnan(psf)] = 0.
            
        img_decon = np.copy(img)

        for n_iter in range(iterations):
            img_decon_last = np.copy(img_decon)
            #derive the foureir transform of the approximated deconvolved image
            img_decon_con = np.fft.rfft2(img_decon)
            #convolve it with the psf
            img_decon_con = img_decon_con * psf
            #and transform it back to the spatial domain
            img_decon_con = np.fft.irfft2(img_decon_con)
            img_decon_con[pad_mask]  = 0.
            
            #derive how far we are off between the deconvolved imaged convolved with the psf and the observed image, i.e., how consistent we are
            deviations = img_decon_con - img
            #and adjust the approximated deconvolved image accordingly
            img_decon -=  k * deviations
            img_decon[img_decon < 0] = 0
            #if the deconvolved image has converged, end the iterations
            dev =  np.max(np.abs(img_decon - img_decon_last))
            if dev <= tolerance[0]: 
                print("Breaking at %d" % n_iter)
                break
        
        #if we are on the gpu, go back to the cpu
        img_decon = cupy.asnumpy(img_decon) 
        
        #undo the padding
        img_decon = img_decon[2048:6144, 2048:6144]
            
        #and we are done
        img_decons.append(img_decon.astype(img.dtype))
    return img_decons


def deconvolve_bid(img, psf, iterations = 25, tolerance = .1, mask = None, use_gpu = True, large_psf = False, pad = True, estimate_background = True, constrain_positive = True):
    """
    Deconvolve an image with the point spread function

    Perform image deconvolution on an image with the instrument
    point spread function using the bid algorithm published in 
    Hofmeister et al. (2024), The Basic Iterative Deconvolution: A Fast Instrumental Point-Spread Function Deconvolution Method That Corrects for Light That Is Scattered Out of the Field of View of a Detector, Solar Physics, Volume 299, Issue 6, article id.77
    https://ui.adsabs.harvard.edu/abs/2023arXiv231211784H/abstract

    Parameters
    ----------
    img : 'numpy 2d array'
        An image.
    psf : `~numpy.ndarray
        The point spread function. 
    iterations: `int`
        Maximum number of iterations
    tolerance: 'float'
        The image deconvolution stops when the maximum change from all pixels between the simulated observed image and the observed image is less than TOLERANCE counts.
    mask: 1d array or 2d array
        Allows to select an image subregion for which the convolution is done. By that, the algorithm can massively speeds up. Can be either a 1d array containing the four elements [left, bottom, right, top], or a 2d array masking the pixels that shall be deconvolved. Actually, if the 2d mask is used, from that the boundaries of the 1d array will be calculated, i.e., all pixels in a corresponding rectangualar box will be deconvolved.
        At the moment, the dimensions of mask has to be smaller than half of the dimensions of the image. If you need to deconvolve a larger region, deconvolve the entire image instead.
    estimate_background: True/False
        If a subregion deconvolution is used, it determines if an incoming scattered light estimate from the surrounding region to the subregion should be applied. This increases the fidelity of the result, but costs some computation time. Generally, it is required for average image intensities and below, but is not required for deconvolving bright image regions.
    use_gpu: True/False
        If True, the deconvolution will be performed on the GPU.
    pad: True/False
        If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
    large_psf: True/False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and deconvolution is done over the full psf.
    constrain_positive: True/False
        Constrain the deconvolution to positive result intensities. If on, it mitigates small ringing artifacts. If true, allow negative intensities in the reconstructions. Negative intensities are informative, as they can tell that something goes wrong (image calibration artifacts, sligthly inaccurate PSF, ringing, etc.)

    Returns
    -------
    `~sunpy.map.Map`
        Deconvolved image

    """
    #At the moment, the mask option only works if the shape of the selected region is smaller than 0.5 * the shape of the image
        
    #if mask is provided as a list, convert it to a 1d-array
    if isinstance(mask, list) or isinstance(mask, tuple): mask = np.array(mask)
    
    #this factor determines the speed of convergence, and should be set between [0.1, 1.0]
    k = 1.
    
    #createa a copy of the image and psf
    img = np.copy(img)
    psf = np.copy(psf)
    
    #for a psf deconvolution, the length of the axis should be even. Thus, if the mask provieded is odd, make it even by adding one row and/or columng
    if isinstance(mask, np.ndarray):
        bd_box_img, bd_box_psf, bd_box_makeeven = get_boundary_boxes(mask, img, psf)
        if estimate_background == True:
            #derive the scattered light of the surrounding into the boundary box, and correct the image for it.
            background = estimate_scattered_light(img, psf, bd_box_img, large_psf = large_psf, pad = pad, use_gpu = use_gpu)        
            img = img - background 
        #cut the image and psf. Since they are cut, large_psf looses its meaning and thus is set to zero
        img = img[bd_box_img[0] : bd_box_img[1] + 1, bd_box_img[2] : bd_box_img[3] + 1]
        psf = psf[bd_box_psf[0] : bd_box_psf[1] + 1, bd_box_psf[2] : bd_box_psf[3] + 1]  
        large_psf = True #If we cut the image, we definitively want to include scattered light over the entire subimage. Thus, large_psf is set to True. The psf boundary box has already been cut before accordingly.

    

    #before the psf deconvolution, we have to pad the image and psf with zeros to break the periodic boundary conditions for the convolution in the fourier domain            
    if pad == True:
        img, psf = pad_img_psf(img, psf, large_psf = large_psf, constant_values = np.nan)
        pad_mask = np.isnan(img)  #the pad mask is required in the next loop - at each iteration, the padding has to be restored.
        img[pad_mask] = 0.
        psf[np.isnan(psf)] = 0.

               
    # if we have a gpu, put it to the gpu
    if use_gpu:
        img = cupy.array(img)
        psf = cupy.array(psf)    
        pad_mask = cupy.array(pad_mask)
        tolerance = cupy.array([tolerance]) 
    else:
        tolerance = [tolerance]
    
    #derive the fourier transform of the psf
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                      psf.shape[1]//2,
                      axis=1)
    psf = np.fft.rfft2(psf) 
        
    img_decon = np.copy(img)
    tolerance = np.array([tolerance])
    for n_iter in range(iterations):
        img_decon_last = np.copy(img_decon)
        #derive the foureir transform of the approximated deconvolved image
        img_decon_con = np.fft.rfft2(img_decon)
        #convolve it with the psf
        img_decon_con = img_decon_con * psf
        #and transform it back to the spatial domain
        img_decon_con = np.fft.irfft2(img_decon_con)
        img_decon_con[pad_mask]  = 0.
        
        #derive how far we are off between the deconvolved imaged convolved with the psf and the observed image, i.e., how consistent we are
        deviations = img_decon_con - img
        #and adjust the approximated deconvolved image accordingly
        img_decon -=  k * deviations
        if constrain_positive == True: img_decon[img_decon < 0] = 0
        #if the deconvolved image has converged, end the iterations
        dev =  np.max(np.abs(img_decon - img_decon_last))
        if dev <= tolerance[0]: break
    
    #if we are on the gpu, go back to the cpu
    if use_gpu:
        img_decon = cupy.asnumpy(img_decon) 
    
    #undo the padding
    if pad == True:
        img_decon, psf = unpad_img_psf(img_decon, psf, large_psf = large_psf)
    #if we had to enlarge the FOV of the mask to get an even pixel length of the axis, shrink the image again                
    if isinstance(mask, list) or isinstance(mask, tuple):
        img_decon = img_decon[0 - bd_box_makeeven[0] : img_decon.shape[0] - bd_box_makeeven[1] + 1,
                              0 - bd_box_makeeven[2] : img_decon.shape[1] - bd_box_makeeven[3] + 1]
        
    #and we are done
    img_decon = img_decon.astype(img.dtype)
    return img_decon
           


def deconvolve_richardson_lucy(img, psf, iterations=25, use_gpu = True, pad = True, large_psf = False, psf_min = 0):
    """
    Deconvolve an image with the point spread function

    Perform image deconvolution on an image with the instrument
    point spread function using the Richardson-Lucy deconvolution
    algorithm

    Parameters
    ----------
    img : 'numpy 2d array'
        An image.
    psf : `~numpy.ndarray`
        The point spread function. 
    iterations: `int`
        Number of iterations in the Richardson-Lucy algorithm
    use_gpu: True/False
        If True, the deconvolution will be performed on the GPU.
    pad: True/False
        If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
    large_psf: True/False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and deconvolution is done over the full psf.

    Returns
    -------
    `~sunpy.map.Map`
        Deconvolved image

    Comments:
        Based on the aiapy.deconvolve method, as described in Cheung, M., 2015, *GPU Technology Conference Silicon Valley*, `GPU-Accelerated Image Processing for NASA's Solar Dynamics Observatory <https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s5209-gpu-accelerated+imaging+processing+for+nasa%27s+solar+dynamics+observatory>`_
    """
    img, psf = np.copy(img), np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25*im_size), int(0.5*im_size)
    
    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size +2*padsize_large_psf
                 
    #padding is only required if the PSF is not large_psf. Else, the padding of the image has already be done above in the large_psf block.
    if pad and not large_psf:  
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size +2*padsize_pad
        psf_size = psf_size +2*padsize_pad

    if use_gpu:
        img = cupy.array(img)
        psf = cupy.array(psf)
        
    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                  psf.shape[1]//2,
                  axis=1)
    
    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    psf_conj = psf.conj()

    img_decon = np.copy(img)
    for _ in range(iterations):
        ratio = img/np.fft.irfft2(np.fft.rfft2(img_decon)*psf)
        img_decon = img_decon*np.fft.irfft2(np.fft.rfft2(ratio)*psf_conj)


    if use_gpu:
        img_decon = cupy.asnumpy(img_decon)
    
    if large_psf:
        img_decon = img_decon[padsize_large_psf : im_size - padsize_large_psf, padsize_large_psf : im_size - padsize_large_psf]
    
    if pad and not large_psf:
        img_decon = img_decon[padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad]
                    
    img_decon = img_decon.astype(img.dtype)
    
    return img_decon


def convolve_image(img, psf, use_gpu = False, pad = True, large_psf = False):
    img = np.copy(img)
    psf = np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25*im_size), int(0.5*im_size)
    
    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size +2*padsize_large_psf
    
    if pad and not large_psf:  
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size +2*padsize_pad
        psf_size = psf_size +2*padsize_pad
    
    if use_gpu:
        img = cupy.array(img)
        psf = cupy.array(psf)
        
    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                  psf.shape[1]//2,
                  axis=1)
    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    img_con = np.fft.rfft2(img)
    img_con = img_con * psf
    img_con = np.fft.irfft2(img_con)

    if use_gpu:
        img_con = cupy.asnumpy(img_con)
        
    if large_psf:
        img_con = img_con[padsize_large_psf : im_size - padsize_large_psf, padsize_large_psf : im_size - padsize_large_psf]
        
    if pad and not large_psf:
        img_con = img_con[padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad]

    img_con = img_con.astype(img.dtype)
    return img_con

  

def pad_img_psf(img, psf, large_psf =  False, constant_values = 0.):
    im_size =  np.array(img.shape)
    padsize_pad, padsize_large_psf = (0.25*im_size).astype(int), (0.5*im_size).astype(int)
    if large_psf:
        img = np.pad(img, ((padsize_large_psf[0], padsize_large_psf[0]), (padsize_large_psf[1], padsize_large_psf[1])), constant_values = constant_values)
        img[img == 0] = 0. #np.finfo(img.dtype).tiny
    else:   
        psf, img = np.pad(psf, ((padsize_pad[0], padsize_pad[0]), (padsize_pad[1], padsize_pad[1])), constant_values = constant_values), np.pad(img, ((padsize_pad[0], padsize_pad[0]), (padsize_pad[1], padsize_pad[1])), constant_values = constant_values)

    return img, psf

def unpad_img_psf(img, psf, large_psf = False):
    im_size =  np.array(img.shape)
    unpadsize_pad, unpadsize_large_psf = (1/6. * im_size).astype(int), (1/4. * im_size).astype(int)
    if large_psf:
        img = img[unpadsize_large_psf[0] : im_size[0] - unpadsize_large_psf[0], unpadsize_large_psf[1] : im_size[1] - unpadsize_large_psf[1]]
    else:
        img = img[unpadsize_pad[0] : im_size[0] - unpadsize_pad[0], unpadsize_pad[1] : im_size[1] - unpadsize_pad[1]]
        psf = psf[unpadsize_pad[0] : im_size[0] - unpadsize_pad[0], unpadsize_pad[1] : im_size[1] - unpadsize_pad[1]]
    return img, psf

def get_boundary_boxes(mask, img, psf):
        if mask.ndim == 1:
            bd_box_img = mask
        if mask.ndim == 2:
            mask = np.where(mask != 0)
            bd_box_img = [ min(mask[0]), max(mask[0]), min(mask[1]), max(mask[1])]
        bd_box_makeeven = np.array([0, 0, 0, 0])
        if (bd_box_img[1] - bd_box_img[0]) %2 == 0:
            bd_box_makeeven[1] = 1
        if (bd_box_img[3] - bd_box_img[2]) %2 == 0:
            bd_box_makeeven[3] = 1 
        if bd_box_img[1] + bd_box_makeeven[1] == img.shape[0]:
            bd_box_makeeven[0] -= 1
            bd_box_makeeven[1] -= 1
        if bd_box_img[3] + bd_box_makeeven[3] == img.shape[1]:
            bd_box_makeeven[2] -= 1
            bd_box_makeeven[3] -= 1
        bd_box_img += bd_box_makeeven
            
        #convert the boundary box of the mask to a corresponding boundary box for the psf
        bd_box_psf = [psf.shape[0]//2 - (bd_box_img[1] - bd_box_img[0] +1), psf.shape[0]//2 + (bd_box_img[1] - bd_box_img[0] +1) -1,
                      psf.shape[1]//2 - (bd_box_img[3] - bd_box_img[2] +1), psf.shape[1]//2 + (bd_box_img[3] - bd_box_img[2] +1) -1]
        return bd_box_img, bd_box_psf, bd_box_makeeven

def estimate_scattered_light(img_in, psf_in, bd_box_img, large_psf = False, pad = True, use_gpu = True):
    #derive the scattered light into the boundary box region
    img = np.copy(img_in)
    psf = np.copy(psf_in)
    
    #as we only want to derive the scattered light from the surrounding into the boundary box, set the image intensity in the boundary box to zero.
    #as we only want to have the scattered light, we set the intrinsic intensity, i.e., the center of the psf, to zero.
    img[bd_box_img[0] : bd_box_img[1] + 1, bd_box_img[2] : bd_box_img[3] + 1] = 0.
    psf[psf.shape[0]//2, psf.shape[1]//2] = 0.

    #pad the image and psf to break the periodic boundary condition involved by the convolution in the fourier domain
    if pad == True:
       img, psf = pad_img_psf(img, psf, large_psf = large_psf)
   
    #put the arrays to the gpu
    if use_gpu:
        img = cupy.array(img)
        psf = cupy.array(psf)    
    
    #derive the fourier transform of the psf
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                      psf.shape[1]//2,
                      axis=1)
    psf = np.fft.rfft2(psf) 
    #derive the foureir transform of the image
    img = np.fft.rfft2(img)
    #convolve it with the psf
    img_background = img * psf
    #and transform it back to the spatial domain
    img_background = np.fft.irfft2(img_background)

    #if we are on the gpu, go back to the cpu
    if use_gpu:
        img_background = cupy.asnumpy(img_background) 
    
    #undo the padding
    if pad == True:
        img_background, psf = unpad_img_psf(img_background, psf, large_psf = large_psf) 
        
    return img_background
    
    
