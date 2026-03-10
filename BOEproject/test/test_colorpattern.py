import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm, MicroLens, ApertureFromImage, CircularAperture

F = PolychromaticField(
    spectrum=1 * cf.illuminant_d65,
    extent_x=10.0 * mm,
    extent_y=10.0 * mm,
    Nx=256*2,
    Ny=256*2,
)
# from microlens_array import create_microlens_array_with_sampling as sag_gen
# sag = 0.001*sag_gen((5,5),10*mm,10*mm,(100,100))
# # 可视化结果
# plt.figure(figsize=(8, 8))
# plt.imshow(sag, extent=([-2.5, 2.5, -2.5, 2.5]), cmap='viridis')
# plt.colorbar(label="Sag (m)")
# plt.title("Microlens Array Sag Profile")
# plt.xlabel("x (mm)")
# plt.ylabel("y (mm)")
# plt.show()


# F.add(MicroLens(5*mm,5*mm,sag,1.5))
# F.add(ApertureFromImage("rectangular_grating.jpg",image_size = (1.2 * mm,1.2 * mm), simulation = F))

F.add(CircularAperture(1*mm))
F.propagate(100*cm)
# F.propagate(1000*cm)
#
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-5*mm, 5*mm], ylim=[-5*mm, 5*mm])
