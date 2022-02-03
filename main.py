# Created by Bastien Rigaud at 02/02/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

from Laplacian import *
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image

def main():
    handle_mask = sitk.ReadImage(r"C:\Data\Data_test\Rectum_ext_0.nii.gz")
    image_mask = sitk.GetArrayFromImage(handle_mask)

    handle_centerline = sitk.ReadImage(r"C:\Data\Data_test\Rectum_ext_0_centerline.nii.gz")
    image_centerline = sitk.GetArrayFromImage(handle_centerline)

    laplacian_filter = Laplacian(input=image_mask, internal=image_centerline, spacing=handle_mask.GetSpacing(),
                                 cl_max=10, cl_min=2, compute_thickness=True, compute_internal_corresp=True,
                                 compute_external_corresp=True)

if __name__ == '__main__':
    main()
