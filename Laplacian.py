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
                 compute_internal_corresp=False, compute_external_corresp=False, verbose=False, padding=2):
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
        :param verbose: show iteration and tau values
        :param padding: padding of the bounding box, default = 2
        '''
        # TODO recover image size
        # convert code to use 4D Numpy?
        # export field images to [x, y, z, c]
        # function to export/save image as sitk with origin and stuff

        if cl_min == 0 or cl_min == cl_max/2:
            raise ValueError("Minimal limit condition cannot be 0 or half the maximal limit condition")

        self.input = input
        self.sx, self.sy, self.sz = spacing
        self.tx = self.ty = self.tz = input.shape
        self.cl_max = cl_max
        self.cl_min = cl_min
        self.val_conv_laplacian = 0.005
        self.conv_rate_thickness = 0.005
        self.conv_rate_corr = 0.000001
        self.iteration_max = 200
        self.verbose = verbose

        original_shape = self.input.shape
        bb_parameters = compute_bounding_box(self.input, padding=padding)
        padding_param =[[bb_parameters[0],original_shape[0]-bb_parameters[1]],
                        [bb_parameters[2],original_shape[1]-bb_parameters[3]],
                        [bb_parameters[4],original_shape[2]-bb_parameters[5]]]
        self.input = self.input[bb_parameters[0]:bb_parameters[1],
                     bb_parameters[2]:bb_parameters[3],
                     bb_parameters[4]:bb_parameters[5]]

        if internal is not None:
            internal = internal[bb_parameters[0]:bb_parameters[1],
                     bb_parameters[2]:bb_parameters[3],
                     bb_parameters[4]:bb_parameters[5]]
        else:
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
            self.l0 = np.pad(self.l0, padding_param)
            self.l1 = np.pad(self.l1, padding_param)
            self.l0_n = np.pad(self.l0_n, padding_param)

        if compute_internal_corresp:
            start_time = time.time()
            self.compute_correspondence_internal()
            print("computed internal correspondence in: {:5.2f} seconds".format(time.time() - start_time))
            self.phi0 = np.pad(self.phi0, padding_param+[[0,0]])

        if compute_external_corresp:
            start_time = time.time()
            self.compute_correspondence_external()
            print("computed external correspondence in: {:5.2f} seconds".format(time.time() - start_time))
            self.phi1 = np.pad(self.phi1, padding_param+[[0,0]])

        self.laplacian = np.pad(self.laplacian*self.input, padding_param)

    def build_model(self, internal=None):
        self.model = self.cl_max * np.ones_like(self.input)
        self.model[self.input > 0] = self.cl_max / 2
        if internal is not None:
            # use a mask of the internal values
            self.model[internal > 0] = self.cl_min
        else:
            # use centroid of the structure
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
            tau = abs((mat_temp - laplacian) / laplacian)
            mat_temp = np.array(laplacian)
            tau_max = np.amax(tau)
            if self.verbose:
                print("it: {}, tau_max = {}".format(iteration, tau_max))
            if tau_max <= self.val_conv_laplacian or iteration == self.iteration_max:
                break
        del mat_temp
        del tau
        return laplacian

    def compute_grad_pix_lp(self):
        self.grad = np.zeros(self.laplacian.shape+(3,), dtype=float)
        self.grad_n = np.zeros(self.laplacian.shape+(3,), dtype=int)
        indices = np.argwhere(self.model == (self.cl_max / 2))
        for l, c, p in indices:
            # compute gradient in the three directions
            self.grad[l, c, p, 0] = (1 / (2 * (self.sx))) * (-self.laplacian[l - 1, c, p] + self.laplacian[l + 1, c, p])
            self.grad[l, c, p, 1] = (1 / (2 * (self.sy))) * (-self.laplacian[l, c - 1, p] + self.laplacian[l, c + 1, p])
            self.grad[l, c, p, 2] = (1 / (2 * (self.sz))) * (-self.laplacian[l, c, p - 1] + self.laplacian[l, c, p + 1])

        # create sign() of gradients
        self.grad_n[self.grad != 0] = self.grad[self.grad != 0] / (abs(self.grad[self.grad != 0]))

        for l, c, p in indices:
            weights = np.sqrt(np.sum(np.square(self.grad[l, c, p, :]), axis=-1))
            self.grad[l, c, p, :] = self.grad[l, c, p, :] / weights

    def compute_thickness(self):
        self.l0 = np.zeros_like(self.laplacian)
        l0_mem = np.zeros_like(self.laplacian)
        l1_mem = np.zeros_like(self.laplacian)
        self.l1 = np.zeros_like(self.laplacian)
        l1_offset = 1000 * np.ones_like(self.laplacian)
        l0_offset = np.zeros_like(self.laplacian)

        self.l1[self.input != 1] = 1000
        l0_offset[self.input == 1] = 1

        l1_offset[self.model == self.cl_min] = 0
        self.l0[self.model == self.cl_min] = 1

        tau_l0 = np.zeros_like(self.laplacian)
        tau_l1 = np.zeros_like(self.laplacian)

        iteration = 0
        indices = np.argwhere(self.model == (self.cl_max / 2))
        while True:
            for l, c, p in indices:
                weights = (1 / (self.sy * self.sz * abs(self.grad[l, c, p, 0]) + self.sx * self.sz * abs(
                    self.grad[l, c, p, 1]) + self.sx * self.sy * abs(self.grad[l, c, p, 2])))
                # compute thickness between internal and external regions
                self.l0[l, c, p] = weights * (
                            self.sx * self.sy * self.sz + self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.l0[
                        l - self.grad_n[l, c, p, 0], c, p] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.l0[
                                l, c - self.grad_n[l, c, p, 1], p] + self.sx * self.sy * abs(self.grad[l, c, p, 2]) *
                            self.l0[l, c, p - self.grad_n[l, c, p, 2]])
                tau_l0[l, c, p] = abs((l0_mem[l, c, p] - self.l0[l, c, p]) / self.l0[l, c, p])
                # compute thickness between external and internal regions
                self.l1[l, c, p] = weights * (
                            self.sx * self.sy * self.sz + self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.l1[
                        l + self.grad_n[l, c, p, 0], c, p] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.l1[
                                l, c + self.grad_n[l, c, p, 1], p] + self.sx * self.sy * abs(self.grad[l, c, p, 2]) *
                            self.l1[l, c, p + self.grad_n[l, c, p, 2]])
                tau_l1[l, c, p] = abs((l1_mem[l, c, p] - self.l1[l, c, p]) / self.l1[l, c, p])
            iteration += 1
            l0_mem = np.array(self.l0)
            l1_mem = np.array(self.l1)
            tau_max = max(np.amax(tau_l0), np.amax(tau_l1))
            if self.verbose:
                print("it: {}, tau_max = {}".format(iteration, tau_max))
            if tau_max <= self.conv_rate_thickness or iteration == self.iteration_max:
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
        self.phi0 = np.zeros(self.laplacian.shape+(3,), dtype=float)

        indices = np.argwhere(self.model == self.cl_min)
        for l, c, p in indices:
            self.phi0[l, c, p, :] = l * self.sx, c * self.sy, p * self.sz
 
        tau_phi0 = np.zeros_like(self.phi0)
        phi0_mem = np.zeros_like(self.phi0)

        indices = np.argwhere(self.model == (self.cl_max / 2))
        iteration = 0
        while True:
            for l, c, p in indices:
                # point correspondences between internal and external regions on x, y and z
                weights = (1 / (self.sy * self.sz * abs(self.grad[l, c, p, 0]) + self.sx * self.sz * abs(self.grad[l, c, p, 1]) + self.sx * self.sy * abs(self.grad[l, c, p, 2])))
                self.phi0[l, c, p, 0] = weights * (self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.phi0[l - self.grad_n[l, c, p, 0], c, p, 0] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.phi0[l, c - self.grad_n[l, c, p, 1], p, 0] + self.sx * self.sy * abs(self.grad[l, c, p, 2]) * self.phi0[l, c, p - self.grad_n[l, c, p, 2], 0])
                self.phi0[l, c, p, 1] = weights * (self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.phi0[l - self.grad_n[l, c, p, 0], c, p, 1] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.phi0[l, c - self.grad_n[l, c, p, 1], p, 1] + self.sx * self.sy * abs(self.grad[l, c, p, 2]) * self.phi0[l, c, p - self.grad_n[l, c, p, 2], 1])
                self.phi0[l, c, p, 2] = weights * (self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.phi0[l - self.grad_n[l, c, p, 0], c, p, 2] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.phi0[l, c - self.grad_n[l, c, p, 1], p, 2] + self.sx * self.sy * abs(self.grad[l, c, p, 2]) * self.phi0[l, c, p - self.grad_n[l, c, p, 2], 2])
                tau_phi0[l, c, p, :] = abs((phi0_mem[l, c, p, :] - self.phi0[l, c, p, :]) / self.phi0[l, c, p, :])

            iteration += 1
            phi0_mem = np.array(self.phi0)
            tau_max = np.nanmax(tau_phi0)
            if self.verbose:
                print("it: {}, tau_max = {}".format(iteration, tau_max))
            if tau_max <= self.conv_rate_corr or iteration == self.iteration_max:
                break

        del tau_phi0
        del phi0_mem

        self.phi0 = self.phi0 * np.repeat(self.input[..., None], 3, axis=-1)

    def compute_correspondence_external(self):
        self.phi1 = np.zeros(self.laplacian.shape+(3,),dtype=float)

        indices = np.argwhere(self.input != 1)
        for l, c, p in indices:
            self.phi1[l, c, p, :] = l * self.sx, c * self.sy, p * self.sz
            
        tau_phi1 = np.zeros_like(self.phi1)
        phi1_mem = np.zeros_like(self.phi1)

        indices = np.argwhere(self.model == (self.cl_max / 2))
        iteration = 0
        while True:
            for l, c, p in indices:
                weights = (1 / (self.sy * self.sz * abs(self.grad[l, c, p, 0]) + self.sx * self.sz * abs(
                    self.grad[l, c, p, 1]) + self.sx * self.sy * abs(self.grad[l, c, p, 2])))
                # point correspondences between external and internal regions on x, y and z
                self.phi1[l, c, p, 0] = weights * (self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.phi1[
                    l + self.grad_n[l, c, p, 0], c, p, 0] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.phi1[
                                                     l, c + self.grad_n[l, c, p, 1], p, 0] + self.sx * self.sy * abs(
                    self.grad[l, c, p, 2]) * self.phi1[l, c, p + self.grad_n[l, c, p, 2], 0])
                self.phi1[l, c, p, 1] = weights * (self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.phi1[
                    l + self.grad_n[l, c, p, 0], c, p, 1] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.phi1[
                                                     l, c + self.grad_n[l, c, p, 1], p, 1] + self.sx * self.sy * abs(
                    self.grad[l, c, p, 2]) * self.phi1[l, c, p + self.grad_n[l, c, p, 2], 1])
                self.phi1[l, c, p, 2] = weights * (self.sy * self.sz * abs(self.grad[l, c, p, 0]) * self.phi1[
                    l + self.grad_n[l, c, p, 0], c, p, 2] + self.sx * self.sz * abs(self.grad[l, c, p, 1]) * self.phi1[
                                                     l, c + self.grad_n[l, c, p, 1], p, 2] + self.sx * self.sy * abs(
                    self.grad[l, c, p, 2]) * self.phi1[l, c, p + self.grad_n[l, c, p, 2], 2])

                tau_phi1[l, c, p, :] = abs((phi1_mem[l, c, p, :] - self.phi1[l, c, p, :]) / self.phi1[l, c, p, :])

            iteration += 1
            phi1_mem = np.array(self.phi1)
            tau_max = np.nanmax(tau_phi1)
            if self.verbose:
                print("it: {}, tau_max = {}".format(iteration, tau_max))
            if tau_max <= self.conv_rate_corr or iteration == self.iteration_max:
                break

        del tau_phi1
        del phi1_mem

        self.phi1 = self.phi1 * np.repeat(self.input[..., None], 3, axis=-1)

