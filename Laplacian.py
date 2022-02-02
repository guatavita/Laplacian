# Created by Bastien Rigaud at 02/02/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Original work by Voisin Camille and Bastien Rigaud
# Modified by Bastien Rigaud
# Description:

import os
import math
import numpy as np
import SimpleITK as sitk

from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image

def compute_bounding_box(annotation, padding=2):
    '''
    :param annotation: A binary image of shape [# images, # rows, # cols, channels]
    :return: the min and max z, row, and column numbers bounding the image
    '''
    shape = annotation.shape
    indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
    min_slice, max_slice = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[0])
    '''
    Get the row values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    min_row, max_row = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[1])
    '''
    Get the col values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    min_col, max_col = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[2])
    return [min_slice, max_slice, min_row, max_row, min_col, max_col]


def numpy_centroid(array, value=1):
    indices = np.argwhere(array == value)
    centroid = np.sum(indices, axis=0) / len(indices)
    return centroid


class Laplacian(object):
    def __init__(self, input, spacing=(1.0, 1.0, 1.0), cl_max=500, cl_min=10):
        self.input = input
        self.sx, self.sy, self.sz = spacing
        self.tx = self.ty = self.tz = input.shape
        self.cl_max = cl_max
        self.cl_min = cl_min
        self.val_conv_laplacian = 0.000001
        self.conv_rate_thickness = 0.005
        self.conv_rate_corr = 0.000001

        original_shape = self.input.shape
        bb_parameters = compute_bounding_box(self.input, padding=2)
        self.input = self.input[bb_parameters[0]:bb_parameters[1],
                bb_parameters[2]:bb_parameters[3],
                bb_parameters[4]:bb_parameters[5]]
        self.cent_x, self.cent_y, self.cent_z = numpy_centroid(self.input).astype(int)
        self.build_model()
        self.max_tx, self.max_ty, self.max_tz = self.model.shape
        self.laplacian = self.compute_laplacian(self.model)
        self.compute_grad_pix_lp()
        self.compute_thickness()
        self.compute_normalize_l0()
        self.compute_correspondance()

    def build_model(self, internal=None):
        self.model = self.cl_max * np.ones_like(self.input)
        self.model[self.input > 0] = self.cl_max / 2
        if internal:
            self.model[internal > 0] = self.cl_min
        else:
            self.model[self.cent_x, self.cent_y, self.cent_z] = self.cl_min

    def compute_laplacian(self, model):
        laplacian = np.copy(model)
        mat_mem = np.copy(model)
        mat_temp = np.copy(model)
        tau = np.zeros_like(model)
        while True:
            for l in range(self.max_tx):
                for c in range(self.max_ty):
                    for p in range(self.max_tz):
                        if mat_mem[l, c, p] == (self.cl_max / 2):
                            laplacian[l, c, p] = (1 / (2 * (self.sy ** 2 * self.sz ** 2 + self.sx ** 2 * self.sz ** 2 + self.sx ** 2 * self.sy ** 2))) * (((self.sy ** 2 * self.sz ** 2) * (laplacian[l - 1, c, p] + laplacian[l + 1, c, p]) + ((self.sx ** 2 * self.sz ** 2) * (laplacian[l, c + 1, p] + laplacian[l, c - 1, p])) + ((self.sx ** 2 * self.sy ** 2) * (laplacian[l, c, p + 1] + laplacian[l, c, p - 1]))))
                            tau[l, c, p] = abs((mat_temp[l, c, p] - laplacian[l, c, p]) / laplacian[l, c, p])
            mat_temp = np.array(laplacian)
            if np.amax(tau) <= self.val_conv_laplacian:
                break
        del mat_mem
        del mat_temp
        del tau
        return laplacian

    def compute_grad_pix_lp(self):
        self.gradx = np.zeros_like(self.laplacian)
        self.grady = np.zeros_like(self.laplacian)
        self.gradz = np.zeros_like(self.laplacian)
        self.gradx_n = np.zeros_like(self.laplacian)
        self.grady_n = np.zeros_like(self.laplacian)
        self.gradz_n = np.zeros_like(self.laplacian)
        for l in range(self.max_tx):
            for c in range(self.max_ty):
                for p in range(self.max_tz):
                    # compute gradient in the three directions
                    self.gradx[l, c, p] = (1 / (2 * (self.sx))) * (-self.laplacian[l - 1, c, p] + self.laplacian[l + 1, c, p])
                    self.grady[l, c, p] = (1 / (2 * (self.sy))) * (-self.laplacian[l, c - 1, p] + self.laplacian[l, c + 1, p])
                    self.gradz[l, c, p] = (1 / (2 * (self.sz))) * (-self.laplacian[l, c, p - 1] + self.laplacian[l, c, p + 1])
                    # create sign() of gradients
                    if (self.gradx[l, c, p] == 0):
                        self.gradx_n[l, c, p] = 0
                    else:
                        self.gradx_n[l, c, p] = self.gradx[l, c, p] / (abs(self.gradx[l, c, p]))
                    if (self.grady[l, c, p] == 0):
                        self.grady_n[l, c, p] = 0
                    else:
                        self.grady_n[l, c, p] = self.grady[l, c, p] / (abs(self.grady[l, c, p]))
                    if (self.gradz[l, c, p] == 0):
                        self.gradz_n[l, c, p] = 0
                    else:
                        self.gradz_n[l, c, p] = self.gradz[l, c, p] / (abs(self.gradz[l, c, p]))
        # gradients normalization
        for l in range(self.max_tx):
            for c in range(self.max_ty):
                for p in range(self.max_tz):
                    self.gradx[l, c, p] = self.gradx[l, c, p] / (np.sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))
                    self.grady[l, c, p] = self.grady[l, c, p] / (np.sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))
                    self.gradz[l, c, p] = self.gradz[l, c, p] / (np.sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))

    def compute_thickness(self):
        self.l0 = np.zeros_like(self.laplacian)
        l0_mem = np.zeros_like(self.laplacian)
        l1_mem = np.zeros_like(self.laplacian)
        self.l1 = np.zeros_like(self.laplacian)
        l1_offset = 1000*np.ones_like(self.laplacian)
        l0_offset = np.zeros_like(self.laplacian)
        tau_l0 = np.zeros_like(self.laplacian)
        tau_l1 = np.zeros_like(self.laplacian)

        self.l1[self.input !=1]=1000
        l0_offset[self.input ==1]=1
        l1_offset[self.cent_x, self.cent_y, self.cent_z] = 0
        self.l0[self.cent_x, self.cent_y, self.cent_z] = 1

        # compute thickness between internal and external regions
        while True:
            for l in range(self.max_tx):
                for c in range(self.max_ty):
                    for p in range(self.max_tz):
                        if self.laplacian[l, c, p] == self.cl_max/2:
                            self.l0[l, c, p] = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sx * self.sy * self.sz + self.sy * self.sz * abs(self.gradx[l, c, p]) * self.l0[l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) *self.l0[l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) * self.l0[l, c, p - self.gradz_n[l, c, p]])
                            tau_l0[l, c, p] = abs((l0_mem[l, c, p] - self.l0[l, c, p]) / self.l0[l, c, p])
            l0_mem = np.array(self.l0)
            if np.amax(tau_l0) <= self.conv_rate_thickness:
                break

        # compute thickness between external and internal regions
        while True:
            for l in range(self.max_tx):
                for c in range(self.max_ty):
                    for p in range(self.max_tz):
                        if self.laplacian[l, c, p] == self.cl_max/2:
                            self.l1[l, c, p] = (1/(self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sx * self.sy * self.sz + self.sy * self.sz * abs(self.gradx[l, c, p]) * self.l1[l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) *self.l1[l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) * self.l1[l, c, p + self.gradz_n[l, c, p]])
                            tau_l1[l, c, p] = abs((l1_mem[l, c, p] - self.l1[l, c, p]) / self.l1[l, c, p])
            l1_mem = np.array(self.l1)
            if np.amax(tau_l1) <= self.conv_rate_thickness:
                break

        self.l0 = self.l0 - l0_offset
        self.l1 = self.l1 - l1_offset
        self.W = self.l1 + self.l0

        del l0_mem
        del l1_mem
        del tau_l0
        del tau_l1
        del l1_offset

    def compute_normalize_l0(self):
        self.l0_n = self.input*(self.l0 / (self.l0 + self.l1))

    def compute_correspondance(self):
        self.phi0x = np.zeros_like(self.laplacian)
        self.phi0y = np.zeros_like(self.laplacian)
        self.phi0z = np.zeros_like(self.laplacian)
        self.phi1x = np.zeros_like(self.laplacian)
        self.phi1y = np.zeros_like(self.laplacian)
        self.phi1z = np.zeros_like(self.laplacian)

        # TODO multiply by spacing?
        for l in range(self.max_tx):
            for c in range(self.max_ty):
                for p in range(self.max_tz):
                    if self.input[l, c, p] != 1:
                        self.phi1x[l, c, p] = l
                        self.phi1y[l, c, p] = c
                        self.phi1z[l, c, p] = p

        self.phi0x[self.cent_x, self.cent_y, self.cent_z] = self.cent_x
        self.phi0y[self.cent_x, self.cent_y, self.cent_z] = self.cent_y
        self.phi0z[self.cent_x, self.cent_y, self.cent_z] = self.cent_z

        tau_phi0x = np.zeros_like(self.laplacian)
        phi0x_mem = np.zeros_like(self.laplacian)
        tau_phi0y = np.zeros_like(self.laplacian)
        phi0y_mem = np.zeros_like(self.laplacian)
        tau_phi0z = np.zeros_like(self.laplacian)
        phi0z_mem = np.zeros_like(self.laplacian)

        tau_phi1x = np.zeros_like(self.laplacian)
        phi1x_mem = np.zeros_like(self.laplacian)
        tau_phi1y = np.zeros_like(self.laplacian)
        phi1y_mem = np.zeros_like(self.laplacian)
        tau_phi1z = np.zeros_like(self.laplacian)
        phi1z_mem = np.zeros_like(self.laplacian)

        while True:
            for l in range(self.max_tx):
                for c in range(self.max_ty):
                    for p in range(self.max_tz):
                        if self.laplacian[l, c, p] == self.cl_max/2:
                            # point correspondances between internal and external regions on x, y and z 
                            self.phi0x[l, c, p] = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi0x[l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi0x[l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *self.phi0x[l, c, p - self.gradz_n[l, c, p]])
                            tau_phi0x[l, c, p] = abs((phi0x_mem[l, c, p] - self.phi0x[l, c, p]) / self.phi0x[l, c, p])
                            self.phi0y[l, c, p] = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi0y[l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi0y[l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *self.phi0y[l, c, p - self.gradz_n[l, c, p]])
                            tau_phi0y[l, c, p] = abs((phi0y_mem[l, c, p] - self.phi0y[l, c, p]) / self.phi0y[l, c, p])
                            self.phi0z[l, c, p] = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi0z[l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi0z[l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *self.phi0z[l, c, p - self.gradz_n[l, c, p]])
                            tau_phi0z[l, c, p] = abs((phi0z_mem[l, c, p] - self.phi0z[l, c, p]) / self.phi0z[l, c, p])

                            # point correspondances between external and internal regions on x, y and z 
                            self.phi1x[l, c, p] = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi1x[l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi1x[l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *self.phi1x[l, c, p + self.gradz_n[l, c, p]])
                            tau_phi1x[l, c, p] = abs((phi1x_mem[l, c, p] - self.phi1x[l, c, p]) / self.phi1x[l, c, p])
                            self.phi1y[l, c, p] = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi1y[l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi1y[l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *self.phi1y[l, c, p + self.gradz_n[l, c, p]])
                            tau_phi1y[l, c, p] = abs((phi1y_mem[l, c, p] - self.phi1y[l, c, p]) / self.phi1y[l, c, p])  # taux de convergence de phi1y
                            self.phi1z[l, c, p] = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p]))) * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi1z[l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi1z[l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *self.phi1z[l, c, p + self.gradz_n[l, c, p]])
                            tau_phi1z[l, c, p] = abs((phi1z_mem[l, c, p] - self.phi1z[l, c, p]) / self.phi1z[l, c, p])


            phi0x_mem = np.array(self.phi0x)
            phi0y_mem = np.array(self.phi0y)
            phi0z_mem = np.array(self.phi0z)
            phi1x_mem = np.array(self.phi1x)
            phi1y_mem = np.array(self.phi1y)
            phi1z_mem = np.array(self.phi1z)

            if np.nanmax(tau_phi0x, tau_phi0y, tau_phi0z) <= self.conv_rate_corr and np.nanmax(tau_phi1x, tau_phi1y, tau_phi1z) <= self.conv_rate_corr:
                break

        del tau_phi0x
        del phi0x_mem
        del tau_phi0y
        del phi0y_mem
        del tau_phi0z
        del phi0z_mem
        del tau_phi1x
        del phi1x_mem
        del tau_phi1y
        del phi1y_mem
        del tau_phi1z
        del phi1z_mem
        
        self.phi0x = self.phi0x * self.input
        self.phi0y = self.phi0y * self.input
        self.phi0z = self.phi0z * self.input
        self.phi1x = self.phi1x * self.input
        self.phi1y = self.phi1y * self.input
        self.phi1z = self.phi1z * self.input
        