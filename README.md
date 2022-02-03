# Laplacian

## Table of contents
* [General info](#general-info)
* [Example](#example)
* [Dependencies](#dependencies)
* [References](#references)

## General info
Bastien Rigaud, PhD
Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
Campus de Beaulieu, Université de Rennes 1
35042 Rennes, FRANCE
bastien.rigaud@univ-rennes1.fr

## Example 

```python
def main():
    handle_mask = sitk.ReadImage(r"C:\Data\Data_test\Rectum_ext_0.nii.gz")
    image_mask = sitk.GetArrayFromImage(handle_mask)

    handle_centerline = sitk.ReadImage(r"C:\Data\Data_test\Rectum_ext_0_centerline.nii.gz")
    image_centerline = sitk.GetArrayFromImage(handle_centerline)

    laplacian_filter = Laplacian(input=image_mask, internal=image_centerline, spacing=handle_mask.GetSpacing(),
                                 cl_max=10, cl_min=2, compute_thickness=True, compute_internal_corresp=True,
                                 compute_external_corresp=True)
```

## Dependencies

Run:
```
pip install -r requirements.txt
```

## References
- Jones, S. E., Buchbinder, B. R., & Aharon, I. (2000). Three‐dimensional mapping of cortical thickness using Laplace's equation. Human brain mapping, 11(1), 12-32.
- Yezzi, A. J., & Prince, J. L. (2003). An Eulerian PDE approach for computing tissue thickness. IEEE transactions on medical imaging, 22(10), 1332-1339.
- Rigaud, B., Cazoulat, G., Vedam, S., Venkatesan, A. M., Peterson, C. B., Taku, N., ... & Brock, K. K. (2020). Modeling complex deformations of the sigmoid colon between external beam radiation therapy and brachytherapy images of cervical cancer. International Journal of Radiation Oncology* Biology* Physics, 106(5), 1084-1094.
