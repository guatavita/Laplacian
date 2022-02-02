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
import time

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
    # Get the row values of primary and secondary
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    min_row, max_row = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[1])
    # Get the col values of primary and secondary
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    min_col, max_col = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[2])
    return [min_slice, max_slice, min_row, max_row, min_col, max_col]


def numpy_centroid(array, value=1):
    indices = np.argwhere(array == value)
    centroid = np.sum(indices, axis=0) / len(indices)
    return centroid


class Laplacian(object):
    def __init__(self, input, internal=None, spacing=(1.0, 1.0, 1.0), cl_max=500, cl_min=10, compute_thickness=False,
                 compute_internal_corresp=False, compute_external_corresp=False):
        '''
        Compute laplacian, gradient, thickness and correspondences between internal and external boundary conditions
        on the input. This code use float64 array because the method relies on iterative computation and float32 is
        slower than float64 when used in for loops
        :param input:
        :param spacing:
        :param cl_max:
        :param cl_min:
        :param compute_thickness:
        :param compute_internal_corresp:
        :param compute_external_corresp:
        '''
        # TODO recover image size
        # Try with internal centerline
        # export field images

        self.input = input
        self.sx, self.sy, self.sz = spacing
        self.tx = self.ty = self.tz = input.shape
        self.cl_max = cl_max
        self.cl_min = cl_min
        self.val_conv_laplacian = 0.005
        self.conv_rate_thickness = 0.005
        self.conv_rate_corr = 0.000001
        self.iteration_max = 2000

        original_shape = self.input.shape
        bb_parameters = compute_bounding_box(self.input, padding=2)
        self.input = self.input[bb_parameters[0]:bb_parameters[1],
                     bb_parameters[2]:bb_parameters[3],
                     bb_parameters[4]:bb_parameters[5]]
        self.cent_x, self.cent_y, self.cent_z = numpy_centroid(self.input).astype(int)
        self.build_model(internal)
        self.max_tx, self.max_ty, self.max_tz = self.model.shape
        start_time = time.time()
        self.laplacian = self.compute_laplacian(self.model)
        print("computed laplacian in: {:5.2f} seconds".format(time.time() - start_time))
        start_time = time.time()
        self.compute_grad_pix_lp()
        print("computed gradient in: {:5.2f} seconds".format(time.time() - start_time))

        if compute_thickness:
            start_time = time.time()
            self.compute_thickness()
            print("computed thickness in: {:5.2f} seconds".format(time.time() - start_time))
            start_time = time.time()
            self.compute_normalize_l0()
            print("computed l0 norm in: {:5.2f} seconds".format(time.time() - start_time))

        if compute_internal_corresp:
            start_time = time.time()
            self.compute_correspondence_internal()
            print("computed internal correspondence in: {:5.2f} seconds".format(time.time() - start_time))

        if compute_external_corresp:
            start_time = time.time()
            self.compute_correspondence_external()
            print("computed external correspondence in: {:5.2f} seconds".format(time.time() - start_time))

    def build_model(self, internal=None):
        self.model = self.cl_max * np.ones_like(self.input)
        self.model[self.input > 0] = self.cl_max / 2
        if internal:
            self.model[internal > 0] = self.cl_min
        else:
            self.model[self.cent_x, self.cent_y, self.cent_z] = self.cl_min

    def compute_laplacian(self, model):
        laplacian = np.array(model).astype(float)
        mat_temp = np.array(model).astype(float)
        iteration = 0
        indices = np.argwhere(self.model == (self.cl_max / 2))
        while True:
            for l, c, p in indices:
                laplacian[l, c, p] = (1 / (2 * (
                            self.sy ** 2 * self.sz ** 2 + self.sx ** 2 * self.sz ** 2 + self.sx ** 2 * self.sy ** 2))) * (
                                     ((self.sy ** 2 * self.sz ** 2) * (
                                                 laplacian[l - 1, c, p] + laplacian[l + 1, c, p]) + (
                                                  (self.sx ** 2 * self.sz ** 2) * (
                                                      laplacian[l, c + 1, p] + laplacian[l, c - 1, p])) + (
                                                  (self.sx ** 2 * self.sy ** 2) * (
                                                      laplacian[l, c, p + 1] + laplacian[l, c, p - 1]))))
            iteration += 1
            if iteration > 10:
                tau = abs((mat_temp - laplacian) / laplacian)
                mat_temp = np.array(laplacian)
                tau_max = np.amax(tau)
                print("it: {}, tau_max = {}".format(iteration, tau_max))
                if tau_max <= self.val_conv_laplacian or iteration == self.iteration_max:
                    break
        del mat_temp
        del tau
        return laplacian * self.input

    def compute_grad_pix_lp(self):
        self.gradx = np.zeros_like(self.laplacian)
        self.grady = np.zeros_like(self.laplacian)
        self.gradz = np.zeros_like(self.laplacian)
        self.gradx_n = np.zeros_like(self.laplacian).astype(int)
        self.grady_n = np.zeros_like(self.laplacian).astype(int)
        self.gradz_n = np.zeros_like(self.laplacian).astype(int)
        indices = np.argwhere(self.model == (self.cl_max / 2))
        for l, c, p in indices:
            # compute gradient in the three directions
            self.gradx[l, c, p] = (1 / (2 * (self.sx))) * (-self.laplacian[l - 1, c, p] + self.laplacian[l + 1, c, p])
            self.grady[l, c, p] = (1 / (2 * (self.sy))) * (-self.laplacian[l, c - 1, p] + self.laplacian[l, c + 1, p])
            self.gradz[l, c, p] = (1 / (2 * (self.sz))) * (-self.laplacian[l, c, p - 1] + self.laplacian[l, c, p + 1])

        # create sign() of gradients
        self.gradx_n[self.gradx != 0] = self.gradx[self.gradx != 0] / (abs(self.gradx[self.gradx != 0]))
        self.grady_n[self.grady != 0] = self.grady[self.grady != 0] / (abs(self.grady[self.grady != 0]))
        self.gradz_n[self.gradz != 0] = self.gradz[self.gradz != 0] / (abs(self.gradz[self.gradz != 0]))

        for l, c, p in indices:
            self.gradx[l, c, p] = self.gradx[l, c, p] / (
                np.sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))
            self.grady[l, c, p] = self.grady[l, c, p] / (
                np.sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))
            self.gradz[l, c, p] = self.gradz[l, c, p] / (
                np.sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))

    def compute_thickness(self):
        self.l0 = np.zeros_like(self.laplacian)
        l0_mem = np.zeros_like(self.laplacian)
        l1_mem = np.zeros_like(self.laplacian)
        self.l1 = np.zeros_like(self.laplacian)
        l1_offset = 1000 * np.ones_like(self.laplacian)
        l0_offset = np.zeros_like(self.laplacian)

        self.l1[self.input != 1] = 1000
        l0_offset[self.input == 1] = 1
        l1_offset[self.cent_x, self.cent_y, self.cent_z] = 0
        self.l0[self.cent_x, self.cent_y, self.cent_z] = 1

        tau_l0 = np.zeros_like(self.laplacian)
        tau_l1 = np.zeros_like(self.laplacian)

        iteration = 0
        indices = np.argwhere(self.model == (self.cl_max / 2))
        while True:
            for l, c, p in indices:
                weights = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(
                    self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p])))
                # compute thickness between internal and external regions
                self.l0[l, c, p] = weights * (
                            self.sx * self.sy * self.sz + self.sy * self.sz * abs(self.gradx[l, c, p]) * self.l0[
                        l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.l0[
                                l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *
                            self.l0[l, c, p - self.gradz_n[l, c, p]])
                tau_l0[l, c, p] = abs((l0_mem[l, c, p] - self.l0[l, c, p]) / self.l0[l, c, p])
                # compute thickness between external and internal regions
                self.l1[l, c, p] = weights * (
                            self.sx * self.sy * self.sz + self.sy * self.sz * abs(self.gradx[l, c, p]) * self.l1[
                        l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.l1[
                                l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(self.gradz[l, c, p]) *
                            self.l1[l, c, p + self.gradz_n[l, c, p]])
                tau_l1[l, c, p] = abs((l1_mem[l, c, p] - self.l1[l, c, p]) / self.l1[l, c, p])
            iteration += 1
            if iteration > 10:
                l0_mem = np.array(self.l0)
                l1_mem = np.array(self.l1)
                tau_max = max(np.amax(tau_l0), np.amax(tau_l1))
                print("it: {}, tau_max = {}".format(iteration, tau_max))
                if tau_max <= self.conv_rate_thickness:
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
        self.l0_n = np.zeros_like(self.laplacian)
        indices = np.argwhere(self.model == (self.cl_max / 2))
        for l, c, p in indices:
            self.l0_n[l, c, p] = (self.l0[l, c, p] / (self.l0[l, c, p] + self.l1[l, c, p]))

    def compute_correspondence_internal(self):
        self.phi0x = np.zeros_like(self.laplacian)
        self.phi0y = np.zeros_like(self.laplacian)
        self.phi0z = np.zeros_like(self.laplacian)

        self.phi0x[self.cent_x, self.cent_y, self.cent_z] = self.cent_x * self.sx
        self.phi0y[self.cent_x, self.cent_y, self.cent_z] = self.cent_y * self.sy
        self.phi0z[self.cent_x, self.cent_y, self.cent_z] = self.cent_z * self.sz

        tau_phi0x = np.zeros_like(self.laplacian)
        phi0x_mem = np.zeros_like(self.laplacian)
        tau_phi0y = np.zeros_like(self.laplacian)
        phi0y_mem = np.zeros_like(self.laplacian)
        tau_phi0z = np.zeros_like(self.laplacian)
        phi0z_mem = np.zeros_like(self.laplacian)

        indices = np.argwhere(self.model == (self.cl_max / 2))
        iteration = 0
        while True:
            for l, c, p in indices:
                # point correspondences between internal and external regions on x, y and z
                weights = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(
                    self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p])))
                self.phi0x[l, c, p] = weights * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi0x[
                    l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi0x[
                                                     l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(
                    self.gradz[l, c, p]) * self.phi0x[l, c, p - self.gradz_n[l, c, p]])
                self.phi0y[l, c, p] = weights * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi0y[
                    l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi0y[
                                                     l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(
                    self.gradz[l, c, p]) * self.phi0y[l, c, p - self.gradz_n[l, c, p]])
                self.phi0z[l, c, p] = weights * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi0z[
                    l - self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi0z[
                                                     l, c - self.grady_n[l, c, p], p] + self.sx * self.sy * abs(
                    self.gradz[l, c, p]) * self.phi0z[l, c, p - self.gradz_n[l, c, p]])

                if iteration > 10:
                    tau_phi0x[l, c, p] = abs((phi0x_mem[l, c, p] - self.phi0x[l, c, p]) / self.phi0x[l, c, p])
                    tau_phi0y[l, c, p] = abs((phi0y_mem[l, c, p] - self.phi0y[l, c, p]) / self.phi0y[l, c, p])
                    tau_phi0z[l, c, p] = abs((phi0z_mem[l, c, p] - self.phi0z[l, c, p]) / self.phi0z[l, c, p])

            iteration += 1
            if iteration > 10:
                phi0x_mem = np.array(self.phi0x)
                phi0y_mem = np.array(self.phi0y)
                phi0z_mem = np.array(self.phi0z)

                tau_max = max(np.nanmax(tau_phi0x), np.nanmax(tau_phi0y), np.nanmax(tau_phi0z))
                print("it: {}, tau_max = {}".format(iteration, tau_max))
                if tau_max <= self.conv_rate_corr:
                    break

        del tau_phi0x
        del phi0x_mem
        del tau_phi0y
        del phi0y_mem
        del tau_phi0z
        del phi0z_mem

        self.phi0x = self.phi0x * self.input
        self.phi0y = self.phi0y * self.input
        self.phi0z = self.phi0z * self.input

    def compute_correspondence_external(self):
        self.phi1x = np.zeros_like(self.laplacian)
        self.phi1y = np.zeros_like(self.laplacian)
        self.phi1z = np.zeros_like(self.laplacian)

        for l in range(self.max_tx):
            for c in range(self.max_ty):
                for p in range(self.max_tz):
                    if self.input[l, c, p] != 1:
                        self.phi1x[l, c, p] = l * self.sx
                        self.phi1y[l, c, p] = c * self.sy
                        self.phi1z[l, c, p] = p * self.sz

        tau_phi1x = np.zeros_like(self.laplacian)
        phi1x_mem = np.zeros_like(self.laplacian)
        tau_phi1y = np.zeros_like(self.laplacian)
        phi1y_mem = np.zeros_like(self.laplacian)
        tau_phi1z = np.zeros_like(self.laplacian)
        phi1z_mem = np.zeros_like(self.laplacian)

        indices = np.argwhere(self.model == (self.cl_max / 2))
        iteration = 0
        while True:
            for l, c, p in indices:
                weights = (1 / (self.sy * self.sz * abs(self.gradx[l, c, p]) + self.sx * self.sz * abs(
                    self.grady[l, c, p]) + self.sx * self.sy * abs(self.gradz[l, c, p])))
                # point correspondences between external and internal regions on x, y and z
                self.phi1x[l, c, p] = weights * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi1x[
                    l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi1x[
                                                     l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(
                    self.gradz[l, c, p]) * self.phi1x[l, c, p + self.gradz_n[l, c, p]])
                self.phi1y[l, c, p] = weights * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi1y[
                    l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi1y[
                                                     l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(
                    self.gradz[l, c, p]) * self.phi1y[l, c, p + self.gradz_n[l, c, p]])
                self.phi1z[l, c, p] = weights * (self.sy * self.sz * abs(self.gradx[l, c, p]) * self.phi1z[
                    l + self.gradx_n[l, c, p], c, p] + self.sx * self.sz * abs(self.grady[l, c, p]) * self.phi1z[
                                                     l, c + self.grady_n[l, c, p], p] + self.sx * self.sy * abs(
                    self.gradz[l, c, p]) * self.phi1z[l, c, p + self.gradz_n[l, c, p]])

                if iteration > 10:
                    tau_phi1x[l, c, p] = abs((phi1x_mem[l, c, p] - self.phi1x[l, c, p]) / self.phi1x[l, c, p])
                    tau_phi1y[l, c, p] = abs((phi1y_mem[l, c, p] - self.phi1y[l, c, p]) / self.phi1y[l, c, p])
                    tau_phi1z[l, c, p] = abs((phi1z_mem[l, c, p] - self.phi1z[l, c, p]) / self.phi1z[l, c, p])

            iteration += 1
            if iteration > 10:
                phi1x_mem = np.array(self.phi1x)
                phi1y_mem = np.array(self.phi1y)
                phi1z_mem = np.array(self.phi1z)
                tau_max = max(np.nanmax(tau_phi1x), np.nanmax(tau_phi1y), np.nanmax(tau_phi1z))
                print("it: {}, tau_max = {}".format(iteration, tau_max))
                if tau_max <= self.conv_rate_corr:
                    break

        del tau_phi1x
        del phi1x_mem
        del tau_phi1y
        del phi1y_mem
        del tau_phi1z
        del phi1z_mem

        self.phi1x = self.phi1x * self.input
        self.phi1y = self.phi1y * self.input
        self.phi1z = self.phi1z * self.input
