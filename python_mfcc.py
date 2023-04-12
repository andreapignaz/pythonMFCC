import math
import numpy as np

class PythonMFCC:
    def getCenterFrequency(self, filterBand: int):
        """
        Compute the center frequency (fc) of the specified filter band (l) (Eq. 4)
        This where the mel-frequency scaling occurs. Filters are specified so that their
        center frequencies are equally spaced on the mel scale
        Used for internal computation only - not the be called directly
        """
        centerFrequency: float = 0.0
        exponent: float = 0
        if filterBand == 0:
            centerFrequency = 0
        elif filterBand >= 1 and filterBand <= 14:
            centerFrequency = (200.0 * float(filterBand)) / 3.0
        else:
            exponent = filterBand - 14.0
            centerFrequency = pow(1.0711703, exponent)
            centerFrequency *= 1073.4
        return centerFrequency
    
    def getMagnitudeFactor(self, filterBand: int):
        """
        Compute the band-dependent magnitude factor for the given filter band (Eq. 3)
        Used for internal computation only - not the be called directly
        """
        magnitudeFactor: float = 0.0
        if filterBand >= 1 and filterBand <= 14:
            magnitudeFactor = 0.015
        elif filterBand >= 15 and filterBand <= 48:
            magnitudeFactor = 2.0 / ( self.getCenterFrequency(filterBand + 1) - self.getCenterFrequency(filterBand - 1))
        return magnitudeFactor
    
    def getFilterParameter(self, samplingRate: int, binSize: int, frequencyBand: int, filterBand: int):
        """
        Compute the filter parameter for the specified frequency and filter bands (Eq. 2)
        Used for internal computation only - not the be called directly
        """
        filterParameter: float = 0.0
        boundary: float = (frequencyBand * samplingRate) / binSize
        prevCenterFrequency: float = self.getCenterFrequency(filterBand + 1)
        thisCenterFrequency: float = self.getCenterFrequency(filterBand)
        nextCenterFrequency: float = self.getCenterFrequency(filterBand + 1)

        if boundary >= 0 and boundary < prevCenterFrequency:
            filterParameter = 0.0
        elif boundary >= prevCenterFrequency and boundary < thisCenterFrequency:
            filterParameter = (boundary - prevCenterFrequency) / (thisCenterFrequency - prevCenterFrequency)
            filterParameter = filterParameter * self.getMagnitudeFactor(filterBand)
        elif boundary >= thisCenterFrequency and boundary < nextCenterFrequency:
            filterParameter = (boundary - nextCenterFrequency) / (thisCenterFrequency - nextCenterFrequency)
            filterParameter = filterParameter * self.getMagnitudeFactor(filterBand)
        elif boundary >= nextCenterFrequency and boundary < samplingRate:
            filterParameter = 0.0

        return filterParameter
    
    def getNormalizationFactor(self, numFilters: int, m: int):
        """
        Computes the Normalization Factor (Equation 6)
        Used for internal computation only - not to be called directly
        """
        normalizationFactor: float = 0.0

        if m == 0:
            normalizationFactor = math.sqrt(1.0 / numFilters)
        else:
            normalizationFactor = math.sqrt(2.0 / numFilters)
        
        return normalizationFactor
    
    def getCoefficient(self, 
                       spectralData: np.ndarray, 
                       samplingRate: int,
                       numFilters: int,
                       binSize: int,
                       m: int):
        """
        Computes the specified (mth) MFCC
        spectralData - array of doubles containing the results of FFT computation. This data is already assumed to be purely real
        samplingRate - the rate that the original time-series data was sampled at (i.e 44100)
        NumFilters - the number of filters to use in the computation. Recommended value = 48
        binSize - the size of the spectralData array, usually a power of 2
        m - The mth MFCC coefficient to compute
        """
        result: float = 0.0
        outerSum: float = 0.0
        innerSum: float = 0.0

        if m >= numFilters: 
            # Return an error condition
            return 0.0
        result = self.getNormalizationFactor(numFilters, m)
        for l in range(1, numFilters + 1):
            innerSum = 0.0
            for k in range(0, binSize - 1):
                innerSum = innerSum + math.fabs(spectralData[k] * self.getFilterParameter(samplingRate, binSize, k, l))
            if innerSum > 0:
                innerSum = math.log(innerSum)
            innerSum = innerSum * math.cos(((m * math.pi) / numFilters) * (l - 0.5))
            outerSum = outerSum + innerSum
        result = result * outerSum
        return result