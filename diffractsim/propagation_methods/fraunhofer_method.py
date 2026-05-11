import numpy as np
from ..util.backend_functions import backend as bd
from ..util.constants import *

def fraunhofer_method(simulation, E, z, λ):
    global bd
    from ..util.backend_functions import backend as bd


    """Fraunhofer diffraction.

    Args:
        u: complex field, shape [H, W] or [B, 1, H, W]
        z: propagation distance
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        n: refractive index
        padding: padding or not

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]

    Reference:
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.5.
    """
    ps = simulation.dx0
    # print(ps)
    Wimg, Himg = E.shape

    # side length
    wvln = λ
    k = 2 * bd.pi / wvln

    # Compute x, y, fx, fy
    L2 = wvln * z / ps

    x2, y2 = bd.meshgrid(
        bd.linspace(-L2 / 2, L2 / 2, Wimg, device=E.device),
        bd.linspace(-L2 / 2, L2 / 2, Himg, device=E.device),
        indexing="xy",
    )

    # Shorter propagation will not affect final result
    # h_amp = 1 / (1j * wvln * z)
    # h_const_phase = bd.exp(1j * k * z)
    # h_phase = bd.exp(1j * bd.pi / (wvln * z) * (x2**2 + y2**2))
    # h = h_amp * h_const_phase * h_phase
    # u = h * ps**2 * bd.fft.ifftshift(bd.fft.fft2(bd.fft.fftshift(E)))
    h = 1/ wvln/z
    u = h  * bd.fft.ifftshift(bd.fft.fft2(bd.fft.fftshift(E)))

    lambdamin = simulation.λ_list_samples[0]*nm
    # Compute x, y, fx, fy
    L2min = lambdamin * z / ps

    x2min, y2min = bd.meshgrid(
        bd.linspace(-L2min / 2, L2min / 2, Wimg, device=E.device),
        bd.linspace(-L2min / 2, L2min / 2, Himg, device=E.device),
        indexing="xy",
    )

    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    # ---------------------------------------
    # 原始坐标
    # ---------------------------------------
    x = x2[0, :]  # shape: [Wimg]
    y = y2[:, 0]  # shape: [Himg]

    # 原始复场 u : [Himg, Wimg]

    # ---------------------------------------
    # 构造插值器
    # 注意:
    # RegularGridInterpolator 的坐标顺序是 (y, x)
    # ---------------------------------------
    interp_real = RegularGridInterpolator(
        (y, x),
        np.real(u),
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    interp_imag = RegularGridInterpolator(
        (y, x),
        np.imag(u),
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    # ---------------------------------------
    # 目标坐标
    # ---------------------------------------
    points = np.stack(
        [y2min.ravel(), x2min.ravel()],
        axis=-1
    )  # [Himg*Wimg, 2]

    # ---------------------------------------
    # 插值
    # ---------------------------------------
    u_real_interp = interp_real(points).reshape(Himg, Wimg)
    u_imag_interp = interp_imag(points).reshape(Himg, Wimg)

    # 恢复复场
    u_interp = u_real_interp + 1j * u_imag_interp

    # simulation.extent_x = L2
    # simulation.extent_y = L2
    #
    # simulation.dx = L2 / Wimg
    # simulation.dy = L2 / Himg
    #
    # simulation.x = simulation.dx * (bd.arange(Wimg) - Wimg // 2)
    # simulation.y = simulation.dy * (bd.arange(Himg) - Himg // 2)
    # simulation.xx, simulation.yy = bd.meshgrid(simulation.x, simulation.y)

    return u_interp
