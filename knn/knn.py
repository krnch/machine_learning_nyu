import csv
import math
import operator

class knn :
    def __init__(self,bank):
        self.trainFile= bank
        self.attributeVectors =[]
        self.distances=[]
        self.attributeVectors1=[]

    def knnClassify(self,attributeVector,k,p):
        distances=[]
        g=[]
        if p==1:
            g=self.attributeVectors1
        else:
            g=self.attributeVectors

        for p in g:
            distance=0
            for i in range(2):
                distance += pow((float(attributeVector[i]) - float(p[i])), 2)
            edistance = math.sqrt(distance)
            distances.append((p,edistance))
        self.distances=distances
        self.distances.sort(key= operator.itemgetter(1))

        neighbors = []
        for x in range(k):
            neighbors.append(self.distances[x][0])

        return neighbors

    def StoreValues(self):
        file = open(self.trainFile,'r')

        for line in file:
            self.attributeVectors.append(line.strip().split(','))
        file.close()



    def Test(self,testFile,k):
        file = open(testFile,'r')
        correct = 0
        total =0
        correct0=0
        correct1=0
        wrong0=0
        wrong1=0
        for line in file:
            vector = line.strip().lower().split(',')
            classifiedNeighbours = self.knnClassify(vector,k,0)
            majorityVotes={}
            for i in range(len(classifiedNeighbours)):
                classValue=classifiedNeighbours[i][-1]

                if classValue in majorityVotes:
                    majorityVotes[classValue] +=1
                else:
                    majorityVotes[classValue] = 1
            sortedVotes =sorted(majorityVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
            finalClassValue =sortedVotes[0][0]
            total +=1
            if finalClassValue==vector[-1]:
                if(vector[-1]=="1"):
                    correct1+=1
                if(vector[-1]=="0"):
                    correct0+=1

                correct +=1
            else:
                if (vector[-1] == "1"):
                    wrong1 += 1
                if (vector[-1] == "0"):
                    wrong0 += 1

        tn=correct0
        fp=wrong0
        fn=wrong1
        tp=correct1

        tpr=(float(tp)/(tp+fn))
        fpr=(float(fp)/(fp+tn))

        print "accuracy for k = "+ str(k) +" is: "+ str((float(correct)/total)*100)
        print"\n Confusion Matrix:"
        print "total="+str(total)+"\t\t"+"Predicted 0"+"\t\t\t"+ "Predicted 1"
        print "Actual 0" +"\t\t" + str(correct0)+ "\t\t\t\t\t"+str(wrong0)
        print "Actual 1" +"\t\t"+str(wrong1)+"\t\t\t\t\t"+str(correct1)

        print"\n True Positive Rate : "+ str(tpr)
        print" False Positive Rate: "+ str(fpr)

    def findKvalue(self,testFile ,k):
        correct = 0
        total = 0
        file = open(testFile,'r')
        for line in file:
            copyDict = []

            vector = line.strip().lower().split(',')
            for i in self.attributeVectors:
                if(i!=vector):
                    copyDict.append(i)


            self.attributeVectors1 = copyDict
            classifiedNeighbours = self.knnClassify(vector,k,1)

            majorityVotes = {}
            for i in range(len(classifiedNeighbours)):
                classValue = classifiedNeighbours[i][-1]

                if classValue in majorityVotes:
                    majorityVotes[classValue] += 1
                else:
                    majorityVotes[classValue] = 1
            sortedVotes = sorted(majorityVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
            finalClassValue = sortedVotes[0][0]
            total +=1
            if finalClassValue==vector[-1]:
                correct +=1
        print "no of correctly classified values for k="+ str(k)+ " is: "+ str(correct)
        print "accuracy for k = " +str(k)+" is: "+ str((float(correct)/total)*100)




if __name__ == "__main__":
    model =knn("traindata.csv")
    model.StoreValues()
    print "for training set :"
    model.findKvalue("traindata.csv",3)
    model.findKvalue("traindata.csv",9)
    model.findKvalue("traindata.csv",99)

    print "\nfor test set :"
    model.Test("testdata.csv",3)

