import os, math, unittest
import numpy as np

class PNN(object):
    def __init__(self, params={'transferFunction':None, 'inputDim':3}):
        if params['transferFunction'] is None:
            self.transferFunction = lambda x:math.exp(x)
        else:
            self.transferFunction = params['transferFunction']
        self.classes = {}
        self.N = 0
        self.inputDimension = params['inputDim']
        self.scaleParameter = 1.0

    def getVectNorm(self, vect):
        s = math.sqrt(sum([x**2.0 for x in vect]))
        return s

    def scaleVector(self, vect, scale=None):
        if scale is None:
            scale = self.getVectNorm(vect)
        outVect = [x/scale for x in vect]
        return outVect

    def addTrainingVector(self, vect, classKey):
        try:
            assert len(vect)==self.inputDimension
        except AssertionError:
            print "Dimension mismatch in addTrainingVector. This won't end well."
        vectScale = self.getVectNorm(vect)
        if vectScale != 1.0:
            vect = self.scaleVector(vect, scale=vectScale)
        if classKey not in self.classes: 
            self.classes[classKey] = []
        self.classes[classKey].append(vect)
        self.N += 1
        self.scaleParameter = float(math.log(float(self.N))/float(self.N))


    def evaluatePattern(self, pattern):
        output = {}
        try:
            assert len(pattern)==self.inputDimension
        except AssertionError:
            print "Dimension mismatchin evaluatePattern. This won't end well."
        patternScale = self.getVectNorm(pattern)
        if patternScale != 1.0:
            pattern = self.scaleVector(pattern, scale=patternScale)
        for classKey in self.classes:
            output[classKey] = None
            #print "\n========\nDiagnostic: Printing self.classes[classKey]: ", str(self.classes[classKey]), "\n========\n"
            multiplier = np.array(self.classes[classKey])
            Ax = np.dot(multiplier, pattern)
            output[classKey] = sum([self.transferFunction((y-1.0)/(self.scaleParameter**2.0)) for y in Ax])
        if len(output)==0:
            output['dummy'] = -100.0
        return max(output, key=output.get)
    def report(self):
        print "Paul has ", str(len(self.classes)), " classes. The number of sample points in each class is:\n"
        for classKey in self.classes:
            print "Class ", str(classKey), " has ", str(len(self.classes[classKey])), " training vectors inside.\n"
        print "Paul should have in total ", str(self.N), " sample points.\n"
        print "Paul has input dimension ", str(self.inputDimension), "\n"
        print "Paul's scaling parameter is ", str(self.scaleParameter), "\n"

        
def class_one_x():
    return np.random.normal(loc=2.0, scale=0.1)
def class_one_y():
    return np.random.normal(loc=2.5, scale=0.2)
def class_two_x():
    return np.random.normal(loc=1.5, scale=1.0)
def class_two_y():
    return np.random.normal(loc=3.0,scale=1.0)

class Test_PNN(unittest.TestCase):
    def test_paul(self):
	paul = PNN(params={'transferFunction':None, 'inputDim':2})
        for i in range(300):
            paul.addTrainingVector([class_one_x(), class_one_y()], classKey='one')
        for j in range(300):
            paul.addTrainingVector([class_two_x(), class_two_y()], classKey='two')
        #print paul.report()
        test_set = []
        for k in range(30):
            u = np.random.randint(low=0,high=2)
            if u==0:
                test_set.append([[class_one_x(), class_one_y()], 'one'])
            else:
                test_set.append([[class_two_x(), class_two_y()], 'two'])
        miscount = 0
        for test_vect in test_set:
            result = paul.evaluatePattern(test_vect[0])
            if  result != test_vect[1]:
                miscount += 1
                print "evaluated pattern to be ", str(result), " but should have gotten ", str(test_vect[1]), "\n"
        errorRate = float(miscount)/float(30)
        print "Misclassified ", str(miscount), " out of 30. That is a ", str(errorRate), " error rate." 

suite = unittest.TestLoader().loadTestsFromTestCase(Test_PNN)
unittest.TextTestRunner(verbosity=1).run(suite)
