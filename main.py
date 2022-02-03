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
    img_pointer = sitk.ReadImage(r"C:\Data\Data_test\Prostate_0.nii.gz")
    img_array = sitk.GetArrayFromImage(img_pointer)
    laplacian_filter = Laplacian(input=img_array, spacing= img_pointer.GetSpacing(), cl_max=2, cl_min=1,
                                 compute_thickness=False, compute_internal_corresp=False,
                                 compute_external_corresp=True)

    xxx = 1

if __name__ == '__main__':
    main()
