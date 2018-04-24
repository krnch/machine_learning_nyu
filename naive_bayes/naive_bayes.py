from __future__ import division
import collections
import math

class NaiveBayes: 
        def __init__(self, phish):
                self.trainFile = phish
                #hardcoding all feature names alongwith the class name -"class" 
                self.attributes = {'URL_Length': ['1', '-1', '0'],
				 'URL_of_Anchor': ['-1', '0', '1'], 
				 'SFH': ['1', '-1', '0'], 
				 'class': ['0', '1', '-1'], 
				 'web_traffic': ['1', '0', '-1'],
				 'age_of_domain': ['1', '-1'], 
				 'SSLfinal_State': ['1', '-1', '0'], 
				 'Request_URL': ['-1', '0', '1'], 
				 'popUpWidnow': ['-1', '0', '1'], 
				 'having_IP_Address': ['0', '1']}   
		#set of feature name lists   
                self.attributeNameList = ['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor', 'web_traffic', 'URL_Length', 'age_of_domain', 'having_IP_Address', 'class']
		# tuple consisting feature counts(class, attribute, attribute_value)      
                self.attributeCounts = collections.defaultdict(lambda: 0.1)
                # entire attribute vector alongwith the class label valuue present in the end
		self.attributeVectors = []       
                self.classLabelCounts = collections.defaultdict(lambda: 0)  
		self.rightPredictions=0
		self.wrongPredictions=0
		self.classProbability = collections.defaultdict(lambda: 0)  


        def Train(self):
                for vector in self.attributeVectors:
			#increment class label
                        self.classLabelCounts[vector[len(vector)-1]] += 1 
                        for counter in range(0, len(vector)-1):
				#increment attribute counts
                                self.attributeCounts[(vector[len(vector)-1], self.attributeNameList[counter], vector[counter])] += 1
		
		for label in self.classLabelCounts:
			self.classProbability[label] = self.classLabelCounts[label]/sum(self.classLabelCounts.values())
			
		#estimated class probability
		print "class probabilities:" 	
		
		print "0 :"+str(self.classProbability['0'])
		print "1 :"+str(self.classProbability['1'])
		print "-1 :"+str(self.classProbability['-1'])

		print "writing all values for P(C) in file probc.txt...."
		for label in self.attributeCounts:
			if(label[0] =='0'):
				
				file=open('probc.txt',"a")
				file.write("P("+label[1]+"="+label[2]+"|C=0)="+str(self.attributeCounts[label]/self.classLabelCounts['0'])+"\n")	
				file.close()	
				
			
			
                #output 
                #denominator for smoothing
		for label in self.classLabelCounts:  
                        for attribute in self.attributeNameList[:len(self.attributeNameList)-1]:
                                self.classLabelCounts[label] += (0.1*len(self.attributes[attribute]))
		
		#answer for question 2
		print "Estimated value of P(xi|C) for C=0 and x2=-1 :"
		print (self.attributeCounts[('0',  'popUpWidnow', '-1')])/(self.classLabelCounts['0'])

        def NaiveClassify(self, attributeVector):      
                probabilityForClass = {}
                for label in self.classLabelCounts:
                        logProb = 0
                        count=0
			
			#skipping mechanism to not consider the last value which is the class label
                        for attributeValue in attributeVector:
				count=count+1;

				if(count==(len(attributeVector))) :
					break
				
                                logProb += math.log(self.attributeCounts[(label, self.attributeNameList[attributeVector.index(attributeValue)], attributeValue)]/self.classLabelCounts[label])
			

                        probabilityForClass[label] = (math.log(self.classLabelCounts[label]/sum(self.classLabelCounts.values())) +logProb)
                return max(probabilityForClass, key = lambda label: probabilityForClass[label])
                                
        def StoreValues(self):
                file = open(self.trainFile, 'r')
                for line in file:        
                	self.attributeVectors.append(line.strip().lower().split(','))                       
                file.close()

        def Test(self, testFile):
                file = open(testFile, 'r')
                for line in file:
                                vector = line.strip().lower().split(',')
				classifiedValue = self.NaiveClassify(vector)	
				if classifiedValue==vector[len(vector)-1]:
					self.rightPredictions += 1
				else:
					self.wrongPredictions += 1

		
if __name__ == "__main__":
        model = NaiveBayes("phish_train.csv")
        model.StoreValues()
        model.Train()
        model.Test("phish_test.csv")
	print "accuracy for test set:"
	print (model.rightPredictions/(model.rightPredictions+model.wrongPredictions))*100
	model.rightPredictions=0
	model.wrongPredictions=0
	model.Test("phish_train.csv")
	print "accuracy for train set:"
	print (model.rightPredictions/(model.rightPredictions+model.wrongPredictions))*100
	
	




