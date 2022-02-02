# Created by Bastien Rigaud at 02/02/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Original work by Voisin Camille and Bastien Rigaud
# Modified by Bastien Rigaud
# Description:

from math import *
from numpy import *
import SimpleITK as sitk
import os

class Laplacian(object):
    def __init__(self):
        init = 1

    def InitPatient(self, Sx, Sy, Sz, Tx, Ty, Tz):
        self.Sx = Sx  # shape coordonnee x
        self.Sy = Sy  # shape coordonnee y
        self.Sz = Sz  # shape coordonnee z
        self.Tx = Tx  # taille image x
        self.Ty = Ty  # taille image y
        self.Tz = Tz  # taille image z

    def BuiltROI(self, Tab_PV):
        self.minTx = self.Tx
        self.maxTx = 0
        self.minTy = self.Ty
        self.maxTy = 0
        self.minTz = self.Tz
        self.maxTz = 0

        # Construction d une box autour de la vessie

        for l in range(0, self.Tx):
            for c in range(0, self.Ty):
                for p in range(0, self.Tz):
                    if (Tab_PV[l, c, p] == 1):
                        if (l < self.minTx):
                            self.minTx = l
                        if (l > self.maxTx):
                            self.maxTx = l
                        if (c < self.minTy):
                            self.minTy = c
                        if (c > self.maxTy):
                            self.maxTy = c
                        if (p < self.minTz):
                            self.minTz = p
                        if (p > self.maxTz):
                            self.maxTz = p

        print("min tx = " + str(self.minTx) + " et maxtx = " + str(self.maxTx))
        print("min ty = " + str(self.minTy) + " et maxty = " + str(self.maxTy))
        print("min tz = " + str(self.minTz) + " et maxtz = " + str(self.maxTz))

        # Marge sur la box

        self.minTx = self.minTx - 2
        self.minTy = self.minTy - 2
        self.minTz = self.minTz - 2
        self.maxTx = self.maxTx + 2
        self.maxTy = self.maxTy + 2
        self.maxTz = self.maxTz + 2

    def Centroid(self, Tab_PV):
        nb_pt = sumx = sumy = sumz = 0
        for l in range(self.minTx, self.maxTx):  # parcours lignes
            for c in range(self.minTy, self.maxTy):  # parcours colonnes
                for p in range(self.minTz, self.maxTz):  # parcours profondeurs
                    if (Tab_PV[l, c, p] == 1):
                        nb_pt += 1
                        sumx += l
                        sumy += c
                        sumz += p

        # Calcul du barycentre de la vessie comme condition interne

        self.cent_X = (sumx / nb_pt)
        self.cent_Y = (sumy / nb_pt)
        self.cent_Z = (sumz / nb_pt)

        print(str(self.cent_X) + "   " + str(self.cent_Y) + "     " + str(self.cent_Z))

    def BuildModel(self, Tab_PV, Clmax, Clmin):
        self.Model = array(Tab_PV, dtype=float)
        for l in range(self.minTx, self.maxTx):  # parcours lignes
            for c in range(self.minTy, self.maxTy):  # parcours colonnes
                for p in range(self.minTz, self.maxTz):  # parcours profondeurs
                    if (Tab_PV[l, c, p] == 1):
                        self.Model[l, c, p] = (Clmax / 2)  # Initialisation des valeurs interne de la vessie ˆ CLmax/2
                    else:
                        self.Model[l, c, p] = Clmax  # CLmax sur l'exterieur de la vessie comme condtion externe fixe

        self.Model[int(self.cent_X), int(self.cent_Y), int(
            self.cent_Z)] = Clmin  # CLmin sur le barycentre comme condtion interne fixe

    def Laplacian(self, Clmax, valconvLp):
        Mat_mem = array(self.Model)  # Memorisation de la matrice avant Lp
        Mat_temp = array(self.Model)  # Creation d'une matrice temp pour calculer difference entre les iterations
        Tau = zeros((self.Tx, self.Ty, self.Tz), dtype=float)  # Condition de convergence entre iterations
        while True:
            for l in range(self.minTx, self.maxTx):  # parcours lignes
                for c in range(self.minTy, self.maxTy):  # parcours colonnes
                    for p in range(self.minTz, self.maxTz):  # parcours profondeurs
                        if (Mat_mem[l, c, p] == Clmax / 2):
                            # Calcul du Laplacien au sein de la region cible
                            self.Model[l, c, p] = (1 / (2 * (
                                        self.Sy ** 2 * self.Sz ** 2 + self.Sx ** 2 * self.Sz ** 2 + self.Sx ** 2 * self.Sy ** 2))) * (
                                                  ((self.Sy ** 2 * self.Sz ** 2) * (
                                                              self.Model[l - 1, c, p] + self.Model[l + 1, c, p]) + (
                                                               (self.Sx ** 2 * self.Sz ** 2) * (
                                                                   self.Model[l, c + 1, p] + self.Model[
                                                               l, c - 1, p])) + ((self.Sx ** 2 * self.Sy ** 2) * (
                                                              self.Model[l, c, p + 1] + self.Model[l, c, p - 1]))))
                            Tau[l, c, p] = abs((Mat_temp[l, c, p] - self.Model[l, c, p]) / self.Model[
                                l, c, p])  # Calcul du taux de convergence
            Mat_temp = array(self.Model)
            if (amax(Tau) <= valconvLp):  # Critere d arret
                break

        # Destruction des anciennes matrices
        del Mat_mem
        del Mat_temp
        del Tau

    def GradPixLp(self):

        # -----------------------------------------------------------------------------------------#
        image_Lp = sitk.ReadImage("/Users/Bastien/Desktop/sauv_13022014/Resultats/Laplacian.mhd")
        self.Model = sitk.GetArrayFromImage(image_Lp)
        # -----------------------------------------------------------------------------------------#

        self.gradx = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.grady = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.gradz = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        self.gradxN = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.gradyN = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.gradzN = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        for l in range(self.minTx, self.maxTx):  # parcours lignes
            for c in range(self.minTy, self.maxTy):  # parcours colonnes
                for p in range(self.minTz, self.maxTz):  # parcours profondeurs

                    # Calcul des gradient dans les 3 directions
                    self.gradx[l, c, p] = (1 / (2 * (self.Sx))) * (-self.Model[l - 1, c, p] + self.Model[l + 1, c, p])
                    self.grady[l, c, p] = (1 / (2 * (self.Sy))) * (-self.Model[l, c - 1, p] + self.Model[l, c + 1, p])
                    self.gradz[l, c, p] = (1 / (2 * (self.Sz))) * (-self.Model[l, c, p - 1] + self.Model[l, c, p + 1])

                    # Creation de sign() des gradients
                    if (self.gradx[l, c, p] == 0):
                        self.gradxN[l, c, p] = 0
                    else:
                        self.gradxN[l, c, p] = self.gradx[l, c, p] / (abs(self.gradx[l, c, p]))
                    if (self.grady[l, c, p] == 0):
                        self.gradyN[l, c, p] = 0
                    else:
                        self.gradyN[l, c, p] = self.grady[l, c, p] / (abs(self.grady[l, c, p]))
                    if (self.gradz[l, c, p] == 0):
                        self.gradzN[l, c, p] = 0
                    else:
                        self.gradzN[l, c, p] = self.gradz[l, c, p] / (abs(self.gradz[l, c, p]))

        # Normalisation des gradients
        for l in range(self.minTx, self.maxTx):  # parcours lignes
            for c in range(self.minTy, self.maxTy):  # parcours colonnes
                for p in range(self.minTz,
                               self.maxTz):  # parcours profondeur
                    self.gradx[l, c, p] = self.gradx[l, c, p] / (
                        sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))
                    self.grady[l, c, p] = self.grady[l, c, p] / (
                        sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))
                    self.gradz[l, c, p] = self.gradz[l, c, p] / (
                        sqrt((self.gradx[l, c, p] ** 2) + (self.grady[l, c, p] ** 2) + (self.gradz[l, c, p] ** 2)))

        print(self.gradx[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)])
        print(self.grady[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)])
        print(self.gradz[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)])

    def Thickness(self, Tab_PV, valconvTh):

        # Destruction des anciennes matrices
        del self.Model

        self.L0 = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        L0mem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        L1mem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.L1 = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        L1offset = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        L0offset = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        TauL0 = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        TauL1 = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        for l in range(self.minTx, self.maxTx):
            for c in range(self.minTy, self.maxTy):
                for p in range(self.minTz, self.maxTz):

                    # Initialisation des epaisseurs
                    if (Tab_PV[l, c, p] != 1):
                        self.L1[l, c, p] = 1000  # en dehors de la vessie
                    else:
                        L0offset[l, c, p] = 1  # offset dans la vessie
                    L1offset[l, c, p] = 1000

        L1offset[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)] = 0
        self.L0[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)] = 1

        # Calcul de l'paisseur de region interne vers externe

        while True:
            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.L0[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                           self.Sx * self.Sy * self.Sz + self.Sy * self.Sz * abs(
                                                       self.gradx[l, c, p]) * self.L0[l - self.gradxN[
                                                       l, c, p], c, p] + self.Sx * self.Sz * abs(self.grady[l, c, p]) *
                                                           self.L0[l, c - self.gradyN[
                                                               l, c, p], p] + self.Sx * self.Sy * abs(
                                                       self.gradz[l, c, p]) * self.L0[l, c, p - self.gradzN[l, c, p]])
                            TauL0[l, c, p] = abs(
                                (L0mem[l, c, p] - self.L0[l, c, p]) / self.L0[l, c, p])  # taux de convergence de L0

            # print(amax(TauL0))
            L0mem = array(self.L0)
            if (amax(TauL0) <= valconvTh):  # Critere d arret
                print("End L0")
                break

        # Calcul de l'paisseur de region externe vers interne

        while True:
            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.L1[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                           self.Sx * self.Sy * self.Sz + self.Sy * self.Sz * abs(
                                                       self.gradx[l, c, p]) * self.L1[l + self.gradxN[
                                                       l, c, p], c, p] + self.Sx * self.Sz * abs(self.grady[l, c, p]) *
                                                           self.L1[l, c + self.gradyN[
                                                               l, c, p], p] + self.Sx * self.Sy * abs(
                                                       self.gradz[l, c, p]) * self.L1[l, c, p + self.gradzN[l, c, p]])
                            TauL1[l, c, p] = abs(
                                (L1mem[l, c, p] - self.L1[l, c, p]) / self.L1[l, c, p])  # taux de convergence de L1

            # print(amax(TauL1))
            L1mem = array(self.L1)
            if (amax(TauL1) <= valconvTh):  # Critere d arret
                print("End L1")
                break

        self.L0 = array(self.L0) - array(L0offset)
        self.L1 = array(self.L1) - array(L1offset)
        self.W = array(self.L1) + array(self.L0)

        # Destruction des anciennes matrices
        del L0mem
        del L1mem
        del TauL0
        del TauL1
        del L1offset

    def NormalizeL0(self, Tab_PV):

        # Normalisation de l epaisseur de la region interne vers externe
        self.L0N = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        for l in range(self.minTx, self.maxTx):
            for c in range(self.minTy, self.maxTy):
                for p in range(self.minTz, self.maxTz):
                    if ((Tab_PV[l, c, p] == 1) & (
                            (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                        self.L0N[l, c, p] = ((self.L0[l, c, p]) / (self.L0[l, c, p] + self.L1[l, c, p]))

    def Connection(self, Tab_PV, valconvTh):

        self.Phi0x = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.Phi0y = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.Phi0z = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.Phi1x = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.Phi1y = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.Phi1z = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        for l in range(self.minTx, self.maxTx):
            for c in range(self.minTy, self.maxTy):
                for p in range(self.minTz, self.maxTz):
                    if (Tab_PV[l, c, p] != 1):
                        self.Phi1x[l, c, p] = l
                        self.Phi1y[l, c, p] = c
                        self.Phi1z[l, c, p] = p

        self.Phi0x[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)] = int(self.cent_X)
        self.Phi0y[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)] = int(self.cent_Y)
        self.Phi0z[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)] = int(self.cent_Z)

        # --------------Phi0x--------------#

        Tauphi0x = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phi0xmem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region interne vers la region externe sur x

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.Phi0x[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                              self.Sy * self.Sz * abs(self.gradx[l, c, p]) * self.Phi0x[
                                                          l - self.gradxN[l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                          self.grady[l, c, p]) * self.Phi0x[l, c - self.gradyN[
                                                          l, c, p], p] + self.Sx * self.Sy * abs(self.gradz[l, c, p]) *
                                                              self.Phi0x[l, c, p - self.gradzN[l, c, p]])
                            Tauphi0x[l, c, p] = abs((Phi0xmem[l, c, p] - self.Phi0x[l, c, p]) / self.Phi0x[
                                l, c, p])  # taux de convergence de phi0x

            Phi0xmem = array(self.Phi0x)
            if (nanmax(Tauphi0x) <= valconvTh):  # Critere d arret
                print("End Phi0x")
                break

        del Tauphi0x
        del Phi0xmem

        # --------------Phi0y--------------#

        Tauphi0y = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phi0ymem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region interne vers la region externe sur y

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.Phi0y[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                              self.Sy * self.Sz * abs(self.gradx[l, c, p]) * self.Phi0y[
                                                          l - self.gradxN[l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                          self.grady[l, c, p]) * self.Phi0y[l, c - self.gradyN[
                                                          l, c, p], p] + self.Sx * self.Sy * abs(self.gradz[l, c, p]) *
                                                              self.Phi0y[l, c, p - self.gradzN[l, c, p]])
                            Tauphi0y[l, c, p] = abs((Phi0ymem[l, c, p] - self.Phi0y[l, c, p]) / self.Phi0y[
                                l, c, p])  # taux de convergence de phi0y

            Phi0ymem = array(self.Phi0y)
            if (nanmax(Tauphi0y) <= valconvTh):  # Critere d arret
                print("End Phi0y")
                break

        del Tauphi0y
        del Phi0ymem

        # --------------Phi0z--------------#

        Tauphi0z = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phi0zmem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region interne vers la region externe sur z

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.Phi0z[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                              self.Sy * self.Sz * abs(self.gradx[l, c, p]) * self.Phi0z[
                                                          l - self.gradxN[l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                          self.grady[l, c, p]) * self.Phi0z[l, c - self.gradyN[
                                                          l, c, p], p] + self.Sx * self.Sy * abs(self.gradz[l, c, p]) *
                                                              self.Phi0z[l, c, p - self.gradzN[l, c, p]])
                            Tauphi0z[l, c, p] = abs((Phi0zmem[l, c, p] - self.Phi0z[l, c, p]) / self.Phi0z[
                                l, c, p])  # taux de convergence de phi0z

            Phi0zmem = array(self.Phi0z)
            if (nanmax(Tauphi0z) <= valconvTh):  # Critere d arret
                print("End Phi0z")
                break

        del Tauphi0z
        del Phi0zmem

        # --------------Phi1x--------------#

        Tauphi1x = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phi1xmem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region externe vers la region interne sur x

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.Phi1x[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                              self.Sy * self.Sz * abs(self.gradx[l, c, p]) * self.Phi1x[
                                                          l + self.gradxN[l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                          self.grady[l, c, p]) * self.Phi1x[l, c + self.gradyN[
                                                          l, c, p], p] + self.Sx * self.Sy * abs(self.gradz[l, c, p]) *
                                                              self.Phi1x[l, c, p + self.gradzN[l, c, p]])
                            Tauphi1x[l, c, p] = abs((Phi1xmem[l, c, p] - self.Phi1x[l, c, p]) / self.Phi1x[
                                l, c, p])  # taux de convergence de phi1x

            Phi1xmem = array(self.Phi1x)
            if (nanmax(Tauphi1x) <= valconvTh):  # Critere d arret
                print("End Phi1x")
                break

        del Tauphi1x
        del Phi1xmem

        # --------------Phi1y--------------#

        Tauphi1y = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phi1ymem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region externe vers la region interne sur y

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.Phi1y[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                              self.Sy * self.Sz * abs(self.gradx[l, c, p]) * self.Phi1y[
                                                          l + self.gradxN[l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                          self.grady[l, c, p]) * self.Phi1y[l, c + self.gradyN[
                                                          l, c, p], p] + self.Sx * self.Sy * abs(self.gradz[l, c, p]) *
                                                              self.Phi1y[l, c, p + self.gradzN[l, c, p]])
                            Tauphi1y[l, c, p] = abs((Phi1ymem[l, c, p] - self.Phi1y[l, c, p]) / self.Phi1y[
                                l, c, p])  # taux de convergence de phi1y

            Phi1ymem = array(self.Phi1y)
            if (nanmax(Tauphi1y) <= valconvTh):  # Critere d arret
                print("End Phi1y")
                break

        del Tauphi1y
        del Phi1ymem

        # --------------Phi1z--------------#

        Tauphi1z = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phi1zmem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region externe vers la region interne sur z

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            self.Phi1z[l, c, p] = (1 / (
                                        self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                    self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                              self.Sy * self.Sz * abs(self.gradx[l, c, p]) * self.Phi1z[
                                                          l + self.gradxN[l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                          self.grady[l, c, p]) * self.Phi1z[l, c + self.gradyN[
                                                          l, c, p], p] + self.Sx * self.Sy * abs(self.gradz[l, c, p]) *
                                                              self.Phi1z[l, c, p + self.gradzN[l, c, p]])
                            Tauphi1z[l, c, p] = abs((Phi1zmem[l, c, p] - self.Phi1z[l, c, p]) / self.Phi1z[
                                l, c, p])  # taux de convergence de phi1z

            Phi1zmem = array(self.Phi1z)
            if (nanmax(Tauphi1z) <= valconvTh):  # Critere d arret
                print("End Phi1z")
                break

        del Tauphi1z
        del Phi1zmem

        # TEST DE LA CORRESPONDANCE

        print("Phi0")
        print("proche barycentre (+10)")
        print(self.Phi0x[int(self.cent_X) + 10, int(self.cent_Y) + 10, int(self.cent_Z) + 10])
        print(self.Phi0y[int(self.cent_X) + 10, int(self.cent_Y) + 10, int(self.cent_Z) + 10])
        print(self.Phi0z[int(self.cent_X) + 10, int(self.cent_Y) + 10, int(self.cent_Z) + 10])

        print("barycentre")
        print(self.Phi0x[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)])
        print(self.Phi0y[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)])
        print(self.Phi0z[int(self.cent_X), int(self.cent_Y), int(self.cent_Z)])

        print("Phi1")
        print("Renvoi coordonnees barycentre ? (+10)")
        print(str(self.cent_X) + "   " + str(self.cent_Y) + "     " + str(self.cent_Z))

        print(self.Phi1x[int(self.cent_X) + 10, int(self.cent_Y) + 10, int(self.cent_Z) + 10])
        print(self.Phi1y[int(self.cent_X) + 10, int(self.cent_Y) + 10, int(self.cent_Z) + 10])
        print(self.Phi1z[int(self.cent_X) + 10, int(self.cent_Y) + 10, int(self.cent_Z) + 10])

        print("+/- 12")
        print(self.Phi1x[int(self.cent_X) + 12, int(self.cent_Y) + 12, int(self.cent_Z) - 12])
        print(self.Phi1y[int(self.cent_X) + 12, int(self.cent_Y) + 12, int(self.cent_Z) - 12])
        print(self.Phi1z[int(self.cent_X) + 12, int(self.cent_Y) + 12, int(self.cent_Z) - 12])

        print("-10")
        print(self.Phi1x[int(self.cent_X) - 10, int(self.cent_Y) - 10, int(self.cent_Z) - 10])
        print(self.Phi1y[int(self.cent_X) - 10, int(self.cent_Y) - 10, int(self.cent_Z) - 10])
        print(self.Phi1z[int(self.cent_X) - 10, int(self.cent_Y) - 10, int(self.cent_Z) - 10])

        self.Phi0x = array(self.Phi0x) * array(Tab_PV)
        self.Phi0y = array(self.Phi0y) * array(Tab_PV)
        self.Phi0z = array(self.Phi0z) * array(Tab_PV)
        self.Phi1x = array(self.Phi1x) * array(Tab_PV)
        self.Phi1y = array(self.Phi1y) * array(Tab_PV)
        self.Phi1z = array(self.Phi1z) * array(Tab_PV)

    def ConnectionSeuil(self, Tab_PV, valconvTh, seuil):

        # -----------------------------------------------------------------------------------------#
        image_L0N = sitk.ReadImage("/Users/Bastien/Desktop/sauv_13022014/Resultats/LongueurL0N.mhd")
        self.L0N = sitk.GetArrayFromImage(image_L0N)
        # -----------------------------------------------------------------------------------------#

        self.Phix = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.Phiy = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        self.Phiz = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        for l in range(self.minTx, self.maxTx):
            for c in range(self.minTy, self.maxTy):
                for p in range(self.minTz, self.maxTz):
                    if ((self.L0N[l, c, p] < seuil + 0.1) & (self.L0N[l, c, p] > seuil - 0.1)):
                        self.Phix[l, c, p] = l
                        self.Phiy[l, c, p] = c
                        self.Phiz[l, c, p] = p

        # --------------Phix--------------#

        Tauphix = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phixmem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region externe vers la region interne sur x par rapport a un seuil sur L0N

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            if (self.L0N[l, c, p] < seuil):  # correspondance externe vers interne
                                # print("in")
                                self.Phix[l, c, p] = (1 / (
                                            self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                        self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                                 self.Sy * self.Sz * abs(self.gradx[l, c, p]) *
                                                                 self.Phix[l + self.gradxN[
                                                                     l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                             self.grady[l, c, p]) * self.Phix[l, c + self.gradyN[
                                                             l, c, p], p] + self.Sx * self.Sy * abs(
                                                             self.gradz[l, c, p]) * self.Phix[
                                                                     l, c, p + self.gradzN[l, c, p]])
                            else:  # correspondance interne vers externe
                                # print("out")
                                self.Phix[l, c, p] = (1 / (
                                            self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                        self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                                 self.Sy * self.Sz * abs(self.gradx[l, c, p]) *
                                                                 self.Phix[l - self.gradxN[
                                                                     l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                             self.grady[l, c, p]) * self.Phix[l, c - self.gradyN[
                                                             l, c, p], p] + self.Sx * self.Sy * abs(
                                                             self.gradz[l, c, p]) * self.Phix[
                                                                     l, c, p - self.gradzN[l, c, p]])

                            # print(self.Phix[l,c,p])
                            Tauphix[l, c, p] = abs((Phixmem[l, c, p] - self.Phix[l, c, p]) / self.Phix[
                                l, c, p])  # taux de convergence de phix

            # print(nanmax(Tauphix))
            Phixmem = array(self.Phix)
            if (nanmax(Tauphix) <= valconvTh):  # Critere d arret
                print("End Phix")
                break

        del Tauphix
        del Phixmem

        # --------------Phiy--------------#

        Tauphiy = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phiymem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region externe vers la region interne sur y par rapport a un seuil sur L0N

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            if (self.L0N[l, c, p] < seuil):  # correspondance externe vers interne
                                self.Phiy[l, c, p] = (1 / (
                                            self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                        self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                                 self.Sy * self.Sz * abs(self.gradx[l, c, p]) *
                                                                 self.Phiy[l + self.gradxN[
                                                                     l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                             self.grady[l, c, p]) * self.Phiy[l, c + self.gradyN[
                                                             l, c, p], p] + self.Sx * self.Sy * abs(
                                                             self.gradz[l, c, p]) * self.Phiy[
                                                                     l, c, p + self.gradzN[l, c, p]])
                            else:  # correspondance interne vers externe
                                self.Phiy[l, c, p] = (1 / (
                                            self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                        self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                                 self.Sy * self.Sz * abs(self.gradx[l, c, p]) *
                                                                 self.Phiy[l - self.gradxN[
                                                                     l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                             self.grady[l, c, p]) * self.Phiy[l, c - self.gradyN[
                                                             l, c, p], p] + self.Sx * self.Sy * abs(
                                                             self.gradz[l, c, p]) * self.Phiy[
                                                                     l, c, p - self.gradzN[l, c, p]])

                            Tauphiy[l, c, p] = abs((Phiymem[l, c, p] - self.Phiy[l, c, p]) / self.Phiy[
                                l, c, p])  # taux de convergence de phiy

            Phiymem = array(self.Phiy)
            if (nanmax(Tauphiy) <= valconvTh):  # Critere d arret
                print("End Phiy")
                break

        del Tauphiy
        del Phiymem

        # --------------Phiz--------------#

        Tauphiz = zeros((self.Tx, self.Ty, self.Tz), dtype=float)
        Phizmem = zeros((self.Tx, self.Ty, self.Tz), dtype=float)

        # Correspondance des points de la region externe vers la region interne sur z par rapport a un seuil sur L0N

        while True:

            for l in range(self.minTx, self.maxTx):
                for c in range(self.minTy, self.maxTy):
                    for p in range(self.minTz, self.maxTz):
                        if ((Tab_PV[l, c, p] == 1) & (
                                (l != int(self.cent_X)) | (c != int(self.cent_Y)) | (p != int(self.cent_Z)))):
                            if (self.L0N[l, c, p] < seuil):  # correspondance externe vers interne
                                self.Phiz[l, c, p] = (1 / (
                                            self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                        self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                                 self.Sy * self.Sz * abs(self.gradx[l, c, p]) *
                                                                 self.Phiz[l + self.gradxN[
                                                                     l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                             self.grady[l, c, p]) * self.Phiz[l, c + self.gradyN[
                                                             l, c, p], p] + self.Sx * self.Sy * abs(
                                                             self.gradz[l, c, p]) * self.Phiz[
                                                                     l, c, p + self.gradzN[l, c, p]])
                            else:  # correspondance interne vers externe
                                self.Phiz[l, c, p] = (1 / (
                                            self.Sy * self.Sz * abs(self.gradx[l, c, p]) + self.Sx * self.Sz * abs(
                                        self.grady[l, c, p]) + self.Sx * self.Sy * abs(self.gradz[l, c, p]))) * (
                                                                 self.Sy * self.Sz * abs(self.gradx[l, c, p]) *
                                                                 self.Phiz[l - self.gradxN[
                                                                     l, c, p], c, p] + self.Sx * self.Sz * abs(
                                                             self.grady[l, c, p]) * self.Phiz[l, c - self.gradyN[
                                                             l, c, p], p] + self.Sx * self.Sy * abs(
                                                             self.gradz[l, c, p]) * self.Phiz[
                                                                     l, c, p - self.gradzN[l, c, p]])

                            Tauphiz[l, c, p] = abs((Phizmem[l, c, p] - self.Phiz[l, c, p]) / self.Phiz[
                                l, c, p])  # taux de convergence de phiz

            Phizmem = array(self.Phiz)
            if (nanmax(Tauphiz) <= valconvTh):  # Critere d arret
                print("End Phiz")
                break

        del Tauphiz
        del Phizmem

        self.Phix = array(self.Phix) * array(Tab_PV)
        self.Phiy = array(self.Phiy) * array(Tab_PV)
        self.Phiz = array(self.Phiz) * array(Tab_PV)

    def Save(self, image_PV, filename):

        # Sauvegarde du Laplacien

        Image = sitk.GetImageFromArray(self.Model)

        Sp = image_PV.GetSpacing()
        O = image_PV.GetOrigin()

        Image.SetSpacing(Sp)
        Image.SetOrigin(O)

        sitk.WriteImage(Image, filename)

    def SaveThickness(self, image_PV, filename0, filename1, filenameW):

        # Sauvegarde des epaisseurs

        Image0 = sitk.GetImageFromArray(self.L0)
        Image1 = sitk.GetImageFromArray(self.L1)
        ImageW = sitk.GetImageFromArray(self.W)

        Sp = image_PV.GetSpacing()
        O = image_PV.GetOrigin()

        Image0.SetSpacing(Sp)
        Image0.SetOrigin(O)

        sitk.WriteImage(Image0, filename0)

        Image1.SetSpacing(Sp)
        Image1.SetOrigin(O)

        sitk.WriteImage(Image1, filename1)

        ImageW.SetSpacing(Sp)
        ImageW.SetOrigin(O)

        sitk.WriteImage(ImageW, filenameW)

    def SaveNormalizeL0(self, image_PV, filenameL0N):

        # Sauvegarde de epaisseur normalisee

        ImageL0N = sitk.GetImageFromArray(self.L0N)

        Sp = image_PV.GetSpacing()
        O = image_PV.GetOrigin()

        ImageL0N.SetSpacing(Sp)
        ImageL0N.SetOrigin(O)

        sitk.WriteImage(ImageL0N, filenameL0N)

    def Savegrad(self, image_PV, filenamex, filenamey, filenamez):

        # Sauvegarde des gradients dans les trois dimensions

        Imagex = sitk.GetImageFromArray(self.gradx)
        Imagey = sitk.GetImageFromArray(self.grady)
        Imagez = sitk.GetImageFromArray(self.gradz)

        Sp = image_PV.GetSpacing()
        O = image_PV.GetOrigin()

        Imagex.SetSpacing(Sp)
        Imagex.SetOrigin(O)

        Imagey.SetSpacing(Sp)
        Imagey.SetOrigin(O)

        Imagez.SetSpacing(Sp)
        Imagez.SetOrigin(O)

        sitk.WriteImage(Imagex, filenamex)
        sitk.WriteImage(Imagey, filenamey)
        sitk.WriteImage(Imagez, filenamez)

    def Savephi(self, image_PV, filename0x, filename0y, filename0z, filename1x, filename1y, filename1z):

        # Sauvegarde des correspondance vers interne/externe

        Image0x = sitk.GetImageFromArray(self.Phi0x)
        Image0y = sitk.GetImageFromArray(self.Phi0y)
        Image0z = sitk.GetImageFromArray(self.Phi0z)
        Image1x = sitk.GetImageFromArray(self.Phi1x)
        Image1y = sitk.GetImageFromArray(self.Phi1y)
        Image1z = sitk.GetImageFromArray(self.Phi1z)

        Sp = image_PV.GetSpacing()
        O = image_PV.GetOrigin()

        Image0x.SetSpacing(Sp)
        Image0x.SetOrigin(O)

        Image0y.SetSpacing(Sp)
        Image0y.SetOrigin(O)

        Image0z.SetSpacing(Sp)
        Image0z.SetOrigin(O)

        Image1x.SetSpacing(Sp)
        Image1x.SetOrigin(O)

        Image1y.SetSpacing(Sp)
        Image1y.SetOrigin(O)

        Image1z.SetSpacing(Sp)
        Image1z.SetOrigin(O)

        sitk.WriteImage(Image0x, filename0x)
        sitk.WriteImage(Image0y, filename0y)
        sitk.WriteImage(Image0z, filename0z)
        sitk.WriteImage(Image1x, filename1x)
        sitk.WriteImage(Image1y, filename1y)
        sitk.WriteImage(Image1z, filename1z)

        del self.Phi0x
        del self.Phi0y
        del self.Phi0z
        del self.Phi1x
        del self.Phi1y
        del self.Phi1z

    def SavePhiSeuil(self, image_PV, filenamex, filenamey, filenamez):

        # Sauvegarde des correspondance pour un seuil fixe

        Imagex = sitk.GetImageFromArray(self.Phix)
        Imagey = sitk.GetImageFromArray(self.Phiy)
        Imagez = sitk.GetImageFromArray(self.Phiz)

        Sp = image_PV.GetSpacing()
        O = image_PV.GetOrigin()

        Imagex.SetSpacing(Sp)
        Imagex.SetOrigin(O)

        Imagey.SetSpacing(Sp)
        Imagey.SetOrigin(O)

        Imagez.SetSpacing(Sp)
        Imagez.SetOrigin(O)

        sitk.WriteImage(Imagex, filenamex)
        sitk.WriteImage(Imagey, filenamey)
        sitk.WriteImage(Imagez, filenamez)

    def SaveText(self, Tab_PV, fileName1, fileName2, seuil):

        # Sauvegarde des coordonnees des correspondances avec un pas de 2

        fic = open(fileName1, 'w')

        for l in range(self.minTx, self.maxTx, 2):
            for c in range(self.minTy, self.maxTy, 2):
                for p in range(self.minTz, self.maxTz, 2):
                    if ((self.L0N[l, c, p] < seuil + 0.1) & (self.L0N[l, c, p] > seuil - 0.1)):
                        fic.write(str(p) + " " + str(c) + " " + str(l) + " ")
                        fic.write(
                            str(self.Phiz[l, c, p]) + " " + str(self.Phiy[l, c, p]) + " " + str(self.Phix[l, c, p]))
                        fic.write('\n')

        fic.close()

        fic = open(fileName2, 'w')

        for l in range(self.minTx, self.maxTx, 2):
            for c in range(self.minTy, self.maxTy, 2):
                for p in range(self.minTz, self.maxTz, 2):
                    if ((self.L0N[l, c, p] < seuil + 0.1) & (self.L0N[l, c, p] > seuil - 0.1)):
                        fic.write(str(l) + " " + str(c) + " " + str(p) + " ")
                        fic.write('\n')
        fic.close()